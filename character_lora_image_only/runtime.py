from contextvars import ContextVar
from functools import partial

import torch

import comfy.ops


SCOPE = "krea2t_character_lora_image_only"
ACTIVE = SCOPE + ".active"
CURRENT = ContextVar(SCOPE, default=None)


def loaded_linear(weight):
    if weight.ndim < 2 or any(size != 1 for size in weight.shape[2:]):
        raise ValueError("Image-only character adapters require linear factors, not spatial convolution kernels.")
    weight = weight.flatten(1)
    layer = comfy.ops.manual_cast.Linear(weight.shape[1], weight.shape[0],
                                         bias=False, device="meta", dtype=weight.dtype)
    layer.weight = torch.nn.Parameter(weight)
    return layer


class Factors(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        up, down, alpha, mid, _, _ = adapter.weights
        self.down = loaded_linear(down)
        self.mid = loaded_linear(mid) if mid is not None else torch.nn.Identity()
        self.up = loaded_linear(up)
        self.scale = 1.0 if alpha is None else float(alpha) / down.shape[0]
        self.in_features = self.down.in_features
        self.out_features = self.up.out_features

    def forward(self, x):
        out = self.up(self.mid(self.down(x)))
        return out if self.scale == 1.0 else out * self.scale


class KroneckerFactors(torch.nn.Module):
    def __init__(self, adapter):
        super().__init__()
        w1, w2, alpha, w1_a, w1_b, w2_a, w2_b, t2, _ = adapter.weights
        rank = None
        if w1 is None:
            self.w1 = torch.nn.Sequential(loaded_linear(w1_b), loaded_linear(w1_a))
            self.groups = w1_b.shape[1]
            out_groups = w1_a.shape[0]
            rank = w1_b.shape[0]
        else:
            self.w1 = loaded_linear(w1)
            self.groups = w1.shape[1]
            out_groups = w1.shape[0]
        if w2 is None:
            layers = [loaded_linear(w2_b)]
            if t2 is None:
                layers.append(loaded_linear(w2_a))
            else:
                layers.extend((loaded_linear(t2), loaded_linear(w2_a.mT)))
            self.w2 = torch.nn.Sequential(*layers)
            in_group, out_group = layers[0].in_features, layers[-1].out_features
            rank = w2_b.shape[0]
        else:
            self.w2 = loaded_linear(w2)
            in_group, out_group = self.w2.in_features, self.w2.out_features
        self.scale = 1.0 if alpha is None or rank is None else float(alpha) / rank
        self.in_features = self.groups * in_group
        self.out_features = out_groups * out_group

    def forward(self, x):
        grouped = x.reshape(*x.shape[:-1], self.groups, -1)
        out = self.w1(self.w2(grouped).transpose(-1, -2)).transpose(-1, -2).flatten(-2)
        return out if self.scale == 1.0 else out * self.scale


class FactorBank(torch.nn.Module):
    def __init__(self, adapters, compute_dtype):
        super().__init__()
        types = {"lora": Factors, "lokr": KroneckerFactors}
        self.factors = torch.nn.ModuleList(types[adapter.name](adapter) for adapter in adapters.values())
        self.manual_cast_dtype = compute_dtype

    def get_dtype(self):
        return self.manual_cast_dtype


def scope_forward(executor, x, timesteps, context, attention_mask=None,
                  ref_latents=None, transformer_options=None, **kwargs):
    options = transformer_options if transformer_options is not None else {}
    token = CURRENT.set({"active": options.get(ACTIVE, ()), "block": None, "rows": None})
    try:
        return executor(x, timesteps, context, attention_mask, ref_latents, options, **kwargs)
    finally:
        CURRENT.reset(token)


class ImageInjection:
    def __init__(self, owner, targets, strength):
        self.owner = owner
        self.targets = targets
        self.strength = strength
        self.handles = []

    def on_model_patcher_clone(self):
        return ImageInjection(self.owner, self.targets, self.strength)

    def before_block(self, index, module, args, kwargs):
        frame = CURRENT.get()
        if frame is None or self.owner not in frame["active"]:
            return
        options = kwargs.get("transformer_options", {})
        bounds = options.get("img_slice")
        if bounds is None or len(bounds) != 2:
            raise RuntimeError("Krea2 did not provide the text/image token boundary for the character LoRA.")
        start, end = map(int, bounds)
        if not 0 <= start <= end == args[0].shape[1]:
            raise RuntimeError("Krea2's text/image token boundary does not match the current block input.")
        frame["block"] = index
        frame["rows"] = start, end

    def add_image_delta(self, index, factor, module, args, output):
        frame = CURRENT.get()
        if frame is None or self.owner not in frame["active"]:
            return output
        if frame["block"] != index or frame["rows"] is None:
            raise RuntimeError("Character LoRA ran outside its shared DiT block scope.")
        start, end = frame["rows"]
        x = args[0]
        if x.ndim != 3 or output.ndim != 3 or x.shape[1] != end or output.shape[1] != end:
            raise RuntimeError("Character LoRA requires Krea2's [batch, tokens, features] linear inputs.")
        if start != end:
            delta = factor(x[:, start:end])
            output[:, start:end].add_(delta.to(dtype=output.dtype), alpha=self.strength)
        return output

    def inject(self, patcher):
        if self.handles:
            return
        bank = patcher.get_additional_models_with_key(self.owner)[0].model
        blocks = set()
        try:
            for key, factor in zip(self.targets, bank.factors):
                parts = key.split(".")
                index = int(parts[2])
                if index not in blocks:
                    block = patcher.model.get_submodule(".".join(parts[:3]))
                    self.handles.append(block.register_forward_pre_hook(
                        partial(self.before_block, index), with_kwargs=True))
                    blocks.add(index)
                module = patcher.model.get_submodule(key.removesuffix(".weight"))
                self.handles.append(module.register_forward_hook(partial(self.add_image_delta, index, factor)))
        except BaseException:
            self.eject()
            raise

    def eject(self):
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()


def inject(patcher, owner):
    patcher.get_attachment(owner).inject(patcher)


def eject(patcher, owner):
    patcher.get_attachment(owner).eject()
