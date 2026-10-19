from contextlib import contextmanager

import math
import torch
import torch.nn.functional as F

from comfy.text_encoders.krea2 import Krea2TEModel, KREA2_TEMPLATE, KREA2_TAP_LAYERS

from .selection import attention_bias, compact_conditioning, sequence_priorities, validate_influence
from .visual_transfer import LAYOUT_KEY, attach_visual_transfer, retained_layout
from .ghost_reference import ghost_reference
from .phrase_parser import parse_phrases
from .phrase_support import phrase_token_indices, remap_phrase_rows


VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


@contextmanager
def encoding_patches(clip, patches):
    inherited = clip.patcher.object_patches.copy()
    try:
        for name, replacement in patches.items():
            clip.patcher.add_object_patch(name, replacement)
        
        yield
    finally:
        clip.patcher.unpatch_model(unpatch_weights=False)
        clip.patcher.object_patches = inherited
        clip.patcher.patch_model(load_weights=False)


def user_prefix_length(tokens):
    rows = tokens["qwen3vl_4b"]
    if len(rows) != 1:
        raise ValueError("Krea2 Virtual Image Conditioning requires one native token sequence.")
    starts = [i for i, row in enumerate(rows[0]) if isinstance(row[0], int) and row[0] == 151644]
    if len(starts) < 2:
        raise ValueError("Missing Krea2 system/user template boundary.")
    end = starts[1] + 3
    if [row[0] for row in rows[0][starts[1] + 1:end]] != [872, 198]:
        raise ValueError("Unexpected Krea2 user template boundary.")
    if any(isinstance(row[0], dict) for row in rows[0][:end]):
        raise ValueError("An image appeared before the user-content boundary.")
    return end


class VisualSelection:

    def __init__(self, references, prefix_length, layer_start, layer_end):
        self.references = references
        self.prefix_length = prefix_length
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.priorities = None
        self.spans = []
        self.masks = {}
        self.report = []

    def capture(self, original):
        def forward(*args, **kwargs):
            embeds = kwargs["embeds"]
            spans = []
            for entry in kwargs["embeds_info"]:
                if entry["type"] == "image":
                    grid = entry["extra"]["grid"].reshape(-1, 3).tolist()
                    if len(grid) != 1:
                        raise ValueError("Each reference must encode as one image grid.")
                    spans.append((entry["index"], entry["size"], tuple(grid[0])))
            self.priorities, self.report = sequence_priorities(embeds.shape[1], spans, self.references)
            self.spans = spans
            self.masks.clear()
            try:
                return original(*args, **kwargs)
            finally:
                self.masks.clear()
        return forward

    def layer(self, original, number):
        def forward(x, attention_mask=None, **kwargs):
            if self.priorities is None:
                raise RuntimeError("Visual span capture did not run before the language encoder.")
            active = self.layer_start <= number <= self.layer_end
            
            key = (id(attention_mask), active, x.device, x.dtype)
            if key not in self.masks:
                bias = attention_bias(self.priorities, active, x.device, x.dtype)
                if bool((bias == 0).all()):
                    self.masks[key] = attention_mask
                else:
                    if attention_mask is None:
                        raise RuntimeError("Expected the native Qwen causal attention mask.")
                    self.masks[key] = attention_mask + bias
            return original(x, attention_mask=self.masks[key], **kwargs)
        return forward

    def encode(self, original):
        def encode_token_weights(tokens):
            
            
            conditioning, pooled, extra = original(tokens, template_end=0)
            conditioning, extra = compact_conditioning(
                conditioning, extra, self.priorities, self.prefix_length,
            )
            extra[LAYOUT_KEY] = retained_layout(self.priorities, self.spans, self.prefix_length)
            self.report.append(f"Output: {conditioning.shape[1]} rows x {conditioning.shape[2]} features.")
            return conditioning, pooled, extra
        return encode_token_weights


class Krea2VirtualImageConditioning:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                   "tooltip": "Use (phrase:weight), e.g. a (red bicycle:1.5). The clean prompt is rendered in the ghost image and encoded as text; phrase weights affect diffusion attention only."}),
            "virtual_image_influence": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 64.0, "step": 0.05,
                "tooltip": "Master priority for the prompt-derived visual reference. "
                           "0 removes all ghost visual rows (vanilla-like); 1 keeps native ghost encoding. "
                           "Higher values prioritize its visual tokens.",
            }),
            "virtual_context_influence": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 64.0, "step": 0.05,
                "tooltip": "Background-versus-text priority INSIDE the ghost image. "
                           "1 keeps the proven original behavior; 0 removes gray background tokens; "
                           "2 doubles the background priority relative to printed prompt regions. "
                           "Requires virtual_image_influence > 0.",
            }),
            "virtual_vision_resolution": ("INT", {
                "default": 0, "min": 0, "max": 2048, "step": 32,
                "tooltip": "Longest side of the prompt-rendered virtual image fed to Qwen vision. "
                           "0 retains the exact image size used by the successful ghost build; "
                           "other values physically resize the image BEFORE native vision preprocessing.",
            }),
            "attention_layer_start": ("INT", {
                "default": 1, "min": 1, "max": 35,
                "tooltip": "First Qwen LANGUAGE layer applying the virtual image priority bias (1-based). "
                           "Layer range matters when virtual_image_influence or context influence differs from 1; "
                           "does not select Qwen's 24 vision blocks.",
            }),
            "attention_layer_end": ("INT", {
                "default": 35, "min": 1, "max": 35,
                "tooltip": "Last Qwen LANGUAGE layer for virtual visual-token bias, inclusive. "
                           "0 visual priorities are blocked in all layers even outside this range.",
            }),
            "diffusion_priority_blocks": ("INT", {
                "default": 8, "min": 0, "max": 28,
                "tooltip": "Apply diffusion virtual-token priority to the first N of Krea2's 28 blocks. "
                           "0 disables. Requires diffusion_priority_strength != 1 or non-neutral "
                           "virtual/context influences. 8 is an inherited, unoptimized default.",
            }),
            "diffusion_priority_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                "tooltip": "Independent multiplier on ghost visual-token attention in diffusion: "
                           "0 disables the diffusion hook, 1 inherits the encoder weights, "
                           "2 doubles their priority even when virtual_image_influence=1.",
            }),
        }}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "virtual_image_report")
    FUNCTION = "encode"
    CATEGORY = "conditioning/krea2"
    DESCRIPTION = "Prompt-rendered virtual image and native Qwen3-VL visual conditioning with inline (phrase:weight) bias on Krea2 diffusion attention. Connect CONDITIONING to the sampler."

    def encode(self, clip, prompt, virtual_image_influence=1.0,
               virtual_context_influence=1.0, virtual_vision_resolution=0,
               attention_layer_start=1, attention_layer_end=35,
               diffusion_priority_blocks=8, diffusion_priority_strength=1.0):
        if not isinstance(clip.cond_stage_model, Krea2TEModel):
            raise TypeError("Load the Qwen3-VL-4B text encoder with CLIPLoader type 'krea2'.")
        stage = clip.cond_stage_model
        inner = getattr(stage, stage.clip)
        layers = inner.transformer.model.layers
        last_tap = max(KREA2_TAP_LAYERS)
        if not 1 <= attention_layer_start <= attention_layer_end <= last_tap:
            raise ValueError(f"Encoder layer range must be ordered within 1..{last_tap}, the last retained hidden state.")
        if not 0 <= diffusion_priority_blocks <= 28:
            raise ValueError("diffusion_priority_blocks must be 0..28")
        if not 0 <= diffusion_priority_strength <= 4 or not math.isfinite(diffusion_priority_strength):
            raise ValueError("diffusion_priority_strength must be finite and within [0, 4]")
        validate_influence(virtual_image_influence)
        validate_influence(virtual_context_influence)
        if not isinstance(virtual_vision_resolution, int) or not 0 <= virtual_vision_resolution <= 2048:
            raise ValueError("virtual_vision_resolution must be 0..2048")

        
        prompt, weighted_spans = parse_phrases(prompt)

        
        ghost, ink_mask = ghost_reference(prompt, with_mask=True)
        if virtual_vision_resolution:
            height, width = ghost.shape[1:3]
            scale = virtual_vision_resolution / max(height, width)
            target_h = max(32, round(height * scale / 32) * 32)
            target_w = max(32, round(width * scale / 32) * 32)
            if (height, width) != (target_h, target_w):
                ghost = F.interpolate(ghost.permute(0, 3, 1, 2), size=(target_h, target_w),
                                      mode="bilinear", align_corners=False, antialias=True).permute(0, 2, 3, 1).contiguous()
                ink_mask = F.interpolate(ink_mask.unsqueeze(1), size=(target_h, target_w),
                                         mode="bilinear", align_corners=False, antialias=True).squeeze(1).contiguous()

        
        images = [ghost]
        references = [(ink_mask, virtual_image_influence,
                       virtual_image_influence * virtual_context_influence, 1)]
        labels = [f"<ghost-reference>{VISION_BLOCK}"]

        
        content = "Reference images:\n" + "\n".join(labels) + "\nEdit instruction:\n" + prompt
        tokens = clip.tokenize(content, images=images, llama_template=KREA2_TEMPLATE)
        raw_phrase_weights = phrase_token_indices(
            content, content.index("Edit instruction:\n") + len("Edit instruction:\n"),
            weighted_spans, tokens,
        )
        prefix_length = user_prefix_length(tokens)
        local_clip = clip.clone()
        selection = VisualSelection(references, prefix_length, attention_layer_start, attention_layer_end)
        transformer_path = f"{stage.clip}.transformer"
        patches = {
            "encode_token_weights": selection.encode(local_clip.patcher.get_model_object("encode_token_weights")),
            f"{transformer_path}.forward": selection.capture(local_clip.patcher.get_model_object(f"{transformer_path}.forward")),
        }
        for index in range(len(layers)):
            path = f"{transformer_path}.model.layers.{index}.forward"
            patches[path] = selection.layer(local_clip.patcher.get_model_object(path), index + 1)
        try:
            with encoding_patches(local_clip, patches):
                conditioning = local_clip.encode_from_tokens_scheduled(tokens)
            
            
            row_weights = remap_phrase_rows(
                tokens, raw_phrase_weights, selection.spans, selection.priorities,
                prefix_length, conditioning[0][0].shape[1],
            )
            phrase_active = any(weight != 1.0 for weight in row_weights)
            
            
            conditioning = attach_visual_transfer(
                conditioning, block_count=diffusion_priority_blocks,
                strength=diffusion_priority_strength,
                phrase_row_weights=row_weights if phrase_active else None,
            )
            report = [
                "Virtual image: prompt-rendered visual reference processed by native Qwen3-VL.",
                "Virtual image occupies reference slot 1; there are no external images or masks.",
                f"Virtual image pixels sent to vision: {ghost.shape[2]}x{ghost.shape[1]} (resolution=0 preserves original).",
                f"Virtual image influence: {virtual_image_influence:g}; background relative influence: {virtual_context_influence:g}.",
                f"Language attention layers: {attention_layer_start}..{attention_layer_end}; native 12 output taps retained.",
                *(line.replace("image_1:", "virtual_image:") for line in selection.report),
                "Zero visual priorities: blocked as keys in every language layer and removed before TextFusion.",
                f"Krea2 diffusion joint-attention priority: first {diffusion_priority_blocks} blocks, independent virtual strength {diffusion_priority_strength:g} (0=off, 1=original). Q/K/V stay unmodified.",
                "No VAE reference latents are added by this node.",
                f"Phrase weighting active: {phrase_active}; mapped weighted phrases: {len(weighted_spans)}.",
                "Phrase scores: log(weight) on generated-image queries -> selected text keys across shared DiT blocks; conditioning hook only, no MODEL wiring.",
            ]
            return conditioning, "\n".join(report)
        finally:
            selection.masks.clear()
            selection.priorities = None
            selection.references = []


NODE_CLASS_MAPPINGS = {"Krea2EditConditioning": Krea2VirtualImageConditioning}
NODE_DISPLAY_NAME_MAPPINGS = {"Krea2EditConditioning": "Krea2 — Virtual Image Conditioning"}
