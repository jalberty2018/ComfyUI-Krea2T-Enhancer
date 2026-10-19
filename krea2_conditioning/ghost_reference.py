import math

import torch
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    result = []
    for source in text.splitlines() or [""]:
        if not source.strip():
            result.append("")
            continue
        line = ""
        for word in source.split():
            candidate = f"{line} {word}" if line else word
            if line and draw.textlength(candidate, font=font) > max_width:
                result.append(line)
                line = word
            else:
                line = candidate
            
            while draw.textlength(line, font=font) > max_width:
                lo, hi = 1, len(line)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if draw.textlength(line[:mid], font=font) <= max_width:
                        lo = mid
                    else:
                        hi = mid - 1
                if lo == 0:
                    raise ValueError("Cannot render a prompt glyph at this font size.")
                result.append(line[:lo])
                line = line[lo:]
        if line:
            result.append(line)
    return result


def ghost_reference(prompt, *, with_mask=False):
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    content = prompt.strip()
    if not content:
        raise ValueError("Text-only ghost conditioning requires a non-empty prompt.")

    width, pad, line_height = 768, 24, 32
    font = _font(24)
    probe = Image.new("RGB", (width, 64), (128, 128, 128))
    draw = ImageDraw.Draw(probe)
    lines = _wrap(draw, content, font, width - pad * 2)
    height = max(128, int(math.ceil((len(lines) * line_height + pad * 2) / 32)) * 32)
    if height > 2048:
        raise ValueError("Prompt too long for the internal ghost-reference image; shorten it or connect an image.")

    canvas = Image.new("RGB", (width, height), (128, 128, 128))
    canvas_draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        canvas_draw.text((pad, pad + i * line_height), line, font=font, fill=(12, 12, 12))
    pixels = torch.from_numpy(__import__("numpy").array(canvas, dtype="float32") / 255.0)
    image = pixels.unsqueeze(0).contiguous()
    if not with_mask:
        return image
    
    
    ink = ((128.0 / 255.0 - pixels[..., 0]) / (116.0 / 255.0)).clamp(0.0, 1.0)
    return image, ink.unsqueeze(0).contiguous()
