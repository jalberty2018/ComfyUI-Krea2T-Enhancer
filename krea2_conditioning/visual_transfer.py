import math

import torch

import comfy.hooks


LAYOUT_KEY = "krea2_edit_visual_layout"
HOOK_ID = "krea2_edit_visual_attention_v2"


def retained_layout(priorities, spans, prefix_length):
    keep = priorities > 0
    keep[:prefix_length] = False
    new_indices = keep.long().cumsum(0) - 1
    rows = []
    for start, size, _grid in spans:
        for old_index in range(start, start + size):
            if bool(keep[old_index]):
                rows.append((int(new_indices[old_index]), float(priorities[old_index])))
    return {"token_count": int(keep.sum()), "visual_rows": tuple(rows)}


class VisualAttentionPriority:
    def __init__(self, layout, first_block=0, block_count=8, strength=1.0, phrase_row_weights=None):
        if not (0 <= first_block <= 27 and 0 <= block_count <= 28 and first_block + block_count <= 28):
            raise ValueError("Invalid Krea2 diffusion block range; valid blocks are 0..27.")
        if not (math.isfinite(strength) and 0 <= strength <= 4):
            raise ValueError("Diffusion priority strength must be between 0 and 4.")
        self.token_count = layout["token_count"]
        
        
        changed = [(row, weight) for row, weight in layout["visual_rows"]
                   if strength > 0 and (weight != 1.0 or strength != 1.0)]
        self.rows = tuple(row for row, _ in changed)
        self.bias = tuple(math.log(weight * strength) for _, weight in changed)
        self.first_block = first_block
        self.last_block = first_block + block_count
        self.visual_enabled = bool(changed and block_count and strength)
        if phrase_row_weights is not None and len(phrase_row_weights) != self.token_count:
            raise ValueError("Phrase weights do not match the final virtual-image conditioning length.")
        
        self.phrase_biases = (tuple(math.log(w) if w > 0 else -80.0
                                   for w in phrase_row_weights)
                               if phrase_row_weights is not None else ())
        self.phrase_enabled = any(w != 0.0 for w in self.phrase_biases)
        self.enabled = self.visual_enabled or self.phrase_enabled

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        options = extra_options or {}
        if options.get("block_type") != "single" or "img_slice" not in options:
            raise ValueError("Krea2 virtual/phrase priorities require native Krea2 joint attention.")
        text_length, total_length = options["img_slice"]
        if text_length != self.token_count or k.shape[-2] != total_length:
            raise ValueError("Visual/phrase row mapping does not match conditioning.")
        block = options.get("block_index")
        use_visual = (self.visual_enabled and block is not None
                      and self.first_block <= block < self.last_block)
        
        use_phrase = self.phrase_enabled
        if not use_visual and not use_phrase:
            return {}

        bias = None
        if use_visual:
            
            bias = torch.zeros((1, 1, 1, total_length), dtype=q.dtype, device=q.device)
            indices = torch.tensor(self.rows, device=q.device, dtype=torch.long)
            bias[..., indices] = torch.tensor(self.bias, dtype=q.dtype, device=q.device)

        if use_phrase:
            
            
            image_end = total_length - sum(options.get("reference_image_num_tokens", ()))
            if not text_length <= image_end <= total_length:
                raise ValueError("Krea2 reference-image tokens disagree with the attention sequence.")
            if image_end > text_length:
                
                
                if bias is None:
                    bias = q.new_zeros((1, 1, total_length, total_length))
                else:
                    bias = bias.expand(1, 1, total_length, total_length).clone()
                bias[..., text_length:image_end, :text_length] += q.new_tensor(self.phrase_biases)

        if bias is None:
            return {}
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                raise ValueError("Cannot safely combine logit priorities with a boolean attention mask.")
            bias = attn_mask + bias
        return {"attn_mask": bias}


def attach_visual_transfer(conditioning, block_count=8, strength=1.0, phrase_row_weights=None):
    result = []
    for tensor, metadata in conditioning:
        layout = metadata[LAYOUT_KEY]
        if tensor.shape[1] != layout["token_count"]:
            raise ValueError("Visual layout metadata does not match returned conditioning.")
        transfer = VisualAttentionPriority(
            layout, block_count=block_count, strength=strength,
            phrase_row_weights=phrase_row_weights,
        )
        if not transfer.enabled:
            result.append([tensor, metadata])
            continue
        hook = comfy.hooks.TransformerOptionsHook(
            {"patches": {"attn1_patch": [transfer]}},
            hook_scope=comfy.hooks.EnumHookScope.HookedOnly,
        )
        hook.hook_id = HOOK_ID
        group = comfy.hooks.HookGroup()
        group.add(hook)
        result.extend(comfy.hooks.set_hooks_for_conditioning([[tensor, metadata]], group))
    return result
