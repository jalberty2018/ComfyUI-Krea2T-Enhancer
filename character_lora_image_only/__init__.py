import json
from pathlib import Path
import uuid

from safetensors import safe_open

import comfy.lora
import comfy.model_patcher
import comfy.utils
import folder_paths
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.patcher_extension import PatcherInjection, WrappersMP

from .runtime import ACTIVE, SCOPE, FactorBank, ImageInjection, eject, inject, scope_forward


PROJECTIONS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate",
               "mlp.gate", "mlp.up", "mlp.down")
TARGETS = frozenset(
    f"diffusion_model.blocks.{block}.{projection}.weight"
    for block in range(28) for projection in PROJECTIONS
)
LORA_PAIRS = (
    (".lora_up.weight", ".lora_down.weight"),
    ("_lora.up.weight", "_lora.down.weight"),
    (".lora_B.weight", ".lora_A.weight"),
    (".lora.up.weight", ".lora.down.weight"),
    (".lora_B", ".lora_A"),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
    (".lora_B.default.weight", ".lora_A.default.weight"),
)


def select_keys(keys, key_map):
    keys = set(keys)
    selected = set()
    for prefix, target in key_map.items():
        if target not in TARGETS or any(prefix + suffix in keys for suffix in (
            ".dora_scale", ".lora_magnitude_vector", ".lora_magnitude_vector.weight",
            ".lora_magnitude_vector.default.weight",
        )):
            continue
        suffixes = []
        for index in (1, 2):
            direct = f".lokr_w{index}"
            pair = (direct + "_a", direct + "_b")
            if prefix + direct in keys:
                suffixes.append(direct)
            elif all(prefix + suffix in keys for suffix in pair):
                suffixes.extend(pair)
            else:
                suffixes = []
                break
        if suffixes:
            if ".lokr_w2_a" in suffixes and prefix + ".lokr_t2" in keys:
                suffixes.append(".lokr_t2")
        else:
            for pair in LORA_PAIRS:
                if all(prefix + suffix in keys for suffix in pair):
                    suffixes = list(pair)
                    if pair == LORA_PAIRS[0] and prefix + ".lora_mid.weight" in keys:
                        suffixes.append(".lora_mid.weight")
                    if prefix + ".reshape_weight" in keys:
                        suffixes.append(".reshape_weight")
                    break
        if suffixes:
            if prefix + ".alpha" in keys:
                suffixes.append(".alpha")
            selected.update(prefix + suffix for suffix in suffixes)
    return sorted(selected)


def read_selected(path, key_map):
    if Path(path).suffix.lower() in (".safetensors", ".sft"):
        with safe_open(path, framework="pt", device="cpu") as source:
            keys = source.keys()
            selected = {key: source.get_tensor(key) for key in select_keys(keys, key_map)}
            return selected, len(keys)
    source = comfy.utils.load_torch_file(path, safe_load=True)
    return {key: source[key] for key in select_keys(source, key_map)}, len(source)


class Krea2TCharacterLoraImageOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"), {
                "tooltip": "Character LoRA or LoKr. Loads supported Krea2 shared DiT projections with their saved scaling; other weights are skipped.",
            }),
            "strength_model": ("FLOAT", {
                "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                "tooltip": "Strength of this character adapter on image-token rows. Text-token rows receive no direct contribution from this adapter.",
            }),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_lora"
    CATEGORY = "Krea2/loaders"
    DESCRIPTION = (
        "Applies a Krea2 character LoRA or LoKr directly to image tokens and skips unsupported weights. "
        "Use in place of the regular loader for that character. Other LoRAs remain active. "
        "Text and image still interact through the model's normal attention."
    )

    @classmethod
    def IS_CHANGED(cls, model, lora_name, strength_model, enabled=True):
        if not enabled or strength_model == 0:
            return "bypassed"
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        stat = Path(path).stat()
        return path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def load_lora(self, model, lora_name, strength_model=1.0, enabled=True):
        report = {"source": lora_name, "filter": "Krea2 shared DiT blocks 0–27; LoRA/LoKr with native key mapping and scaling",
                  "strength_model": strength_model, "scope": "image-token rows, including reference-image rows"}
        if not enabled or strength_model == 0:
            return model, json.dumps(dict(report, status="bypassed"), indent=2)

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        if not isinstance(model.get_model_object("diffusion_model"), SingleStreamDiT):
            raise TypeError("Image-only character LoRA requires the native Krea2 model.")
        key_map = {key: target for key, target in comfy.lora.model_lora_keys_unet(model.model, {}).items()
                   if isinstance(target, str) and target in TARGETS}
        state, source_count = read_selected(path, key_map)
        report.update(source_tensor_keys=source_count, retained_tensor_keys=len(state),
                      skipped_tensor_keys=source_count - len(state))
        if not state:
            return model, json.dumps(dict(report, status="no matching complete LoRA/LoKr adapters"), indent=2)

        adapters = comfy.lora.load_lora(state, key_map, log_missing=False)
        bank = FactorBank(adapters, model.model_dtype())
        for key, factor in zip(adapters, bank.factors):
            module = model.model.get_submodule(key.removesuffix(".weight"))
            if (factor.out_features, factor.in_features) != (module.out_features, module.in_features):
                raise ValueError(f"Character adapter dimensions do not match Krea2 target {key}.")
        factors = comfy.model_patcher.ModelPatcher(bank, model.load_device, model.offload_device)
        patched = model.clone()
        owner = SCOPE + "." + uuid.uuid4().hex
        patched.set_additional_models(owner, [factors])
        patched.set_attachments(owner, ImageInjection(owner, tuple(adapters), strength_model))
        patched.set_injections(owner, [PatcherInjection(
            inject=lambda patcher: inject(patcher, owner),
            eject=lambda patcher: eject(patcher, owner),
        )])
        options = patched.model_options["transformer_options"]
        options[ACTIVE] = (*options.get(ACTIVE, ()), owner)
        patched.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, SCOPE)
        patched.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, SCOPE, scope_forward)
        report.update(status="active", adapted_matrices=len(adapters),
                      adapter_types={name: sum(adapter.name == name for adapter in adapters.values())
                                     for name in sorted({adapter.name for adapter in adapters.values()})},
                      adapter_scale_range=[min(factor.scale for factor in bank.factors),
                                           max(factor.scale for factor in bank.factors)],
                      factor_bytes=factors.model_size(), existing_input_patches="preserved")
        return patched, json.dumps(report, indent=2)


NODE_CLASS_MAPPINGS = {"Krea2TCharacterLoraImageOnly": Krea2TCharacterLoraImageOnly}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2TCharacterLoraImageOnly": "Krea2T Character LoRA — Image Only"}
