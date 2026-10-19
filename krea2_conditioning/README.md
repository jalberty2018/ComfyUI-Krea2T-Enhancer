# Krea2 — Virtual Image Conditioning

Folder: `krea2_conditioning/`.

This encoder creates an internal image containing the clean prompt text and passes it through native Qwen3-VL visual encoding alongside normal prompt encoding. It has no external IMAGE or MODEL sockets.

Connect Krea2 `CLIP` and `prompt`; send `CONDITIONING` to your guider/sampler. Optional `(phrase:weight)` syntax adjusts image-query attention to the corresponding text keys in Krea2. No MODEL wiring is required.

- `virtual_image_influence`: master visual-token influence; `0` suppresses ghost rows, `1` retains the original ghost pathway.
- `virtual_context_influence`: relative weighting for gray-background regions of the internal image (`1` neutral).
- `virtual_vision_resolution`: longest side of the internal image before Qwen preprocessing (`0` retains native ghost dimensions).
- `attention_layer_start` / `attention_layer_end`: language-layer range for nonneutral visual-key bias.
- `diffusion_priority_blocks`: number of initial diffusion blocks receiving visual-key priority (`0` disables the diffusion visual adjustment).
- `diffusion_priority_strength`: additional visual-key priority in those selected diffusion blocks (`1` neutral at native visual weights).

Phrase example: `a (red bicycle:1.5) on a (rainy street:0.7)`. Phrase annotations are removed before image rendering and tokenization.

The virtual image is an image of printed prompt text, not a synthesized picture of the requested scene. The default 8-block diffusion range is experimental, not a validated optimum.

See the repository root `README.md` for integration notes.
