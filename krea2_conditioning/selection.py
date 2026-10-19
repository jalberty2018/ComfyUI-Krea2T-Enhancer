import math

import torch
import torch.nn.functional as F


def validate_influence(value):
    if not math.isfinite(value) or value < 0:
        raise ValueError("Visual influence must be finite and non-negative.")


def region_priorities(mask, grid, selected, context):
    validate_influence(selected)
    validate_influence(context)
    frames, height, width = grid
    if frames != 1 or height % 2 or width % 2:
        raise ValueError(f"Expected one image on a 2x2 merged Qwen vision grid; received {grid}.")
    height, width = height // 2, width // 2
    if mask is None:
        coverage = torch.ones(height * width)
    else:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim != 3 or mask.shape[0] != 1:
            raise ValueError("Each reference mask must contain exactly one image.")
        mask = mask.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError("Reference masks must contain finite values between 0 and 1.")
        
        coverage = F.interpolate(mask.unsqueeze(1), size=(height, width), mode="area").flatten()
    return context + coverage * (selected - context)


def sequence_priorities(length, spans, references):
    if len(spans) != len(references):
        raise ValueError(f"Encoded {len(spans)} image spans for {len(references)} references.")
    priorities = torch.ones(length)
    report = []
    last_end = 0
    for span, reference in zip(spans, references):
        start, size, grid = span
        if start < last_end or start + size > length:
            raise ValueError("Expanded visual spans overlap or exceed the encoded sequence.")
        mask, selected, context, slot = reference
        weights = region_priorities(mask, grid, selected, context)
        if weights.numel() != size:
            raise ValueError(f"Image {slot}: grid maps to {weights.numel()} rows, encoder returned {size}.")
        priorities[start:start + size] = weights
        kept = int(torch.count_nonzero(weights))
        report.append(f"image_{slot}: grid={grid[1] // 2}x{grid[2] // 2}; expanded rows=[{start},{start + size}); kept={kept}/{size}; selected={selected:g}; context={context:g}")
        last_end = start + size
    return priorities, report


def attention_bias(priorities, active, device, dtype):
    weights = priorities.to(device=device, dtype=torch.float32)
    if active:
        return weights.log().to(dtype=dtype)
    return torch.zeros_like(weights, dtype=dtype).masked_fill(weights == 0, -torch.inf)


def compact_conditioning(conditioning, extra, priorities, prefix_length):
    if conditioning.shape[1] != priorities.numel():
        raise ValueError("Conditioning rows no longer match the captured expanded image spans.")
    keep = priorities > 0
    keep[:prefix_length] = False
    if not keep.any():
        raise ValueError("No conditioning rows remain after selection.")
    conditioning = conditioning[:, keep.to(conditioning.device)].contiguous()
    extra = extra.copy()
    if "attention_mask" in extra:
        mask = extra["attention_mask"]
        extra["attention_mask"] = mask[:, keep.to(mask.device)].contiguous()
        if bool(extra["attention_mask"].all()):
            extra.pop("attention_mask")
    return conditioning, extra
