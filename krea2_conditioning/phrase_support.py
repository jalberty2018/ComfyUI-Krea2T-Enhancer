import hashlib
import json
from pathlib import Path

import torch

from comfy.text_encoders.krea2 import KREA2_TEMPLATE
from .phrase_parser import parse_phrases


_TOKEN_KEY = "qwen3vl_4b"
_TOKENIZER = None
_IMAGE_PAD = 151655


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import Qwen2TokenizerFast
        from comfy.text_encoders import krea2
        path = Path(krea2.__file__).parent / "qwen25_tokenizer"
        _TOKENIZER = Qwen2TokenizerFast.from_pretrained(path, local_files_only=True)
    return _TOKENIZER


def phrase_token_indices(content, instruction_offset, phrases, native_tokens, tokenizer=None):
    if not phrases:
        return {}
    tokenizer = tokenizer or _tokenizer()
    full_text = KREA2_TEMPLATE.format(content)
    prefix_text = KREA2_TEMPLATE.split("{}", 1)[0]
    prompt_start = len(prefix_text) + instruction_offset
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    pairs = native_tokens[_TOKEN_KEY][0]
    if len(pairs) != len(token_ids):
        raise ValueError("Qwen multimodal token count differs from the exact phrase token map.")
    for i, (pair, expected) in enumerate(zip(pairs, token_ids)):
        value = pair[0]
        if isinstance(value, dict):
            if value.get("type") != "image" or expected != _IMAGE_PAD:
                raise ValueError(f"Unexpected embedded token at position {i}; cannot map phrase weights.")
        elif not isinstance(value, int) or int(value) != int(expected):
            raise ValueError(f"Krea2 tokenizer mismatch at position {i}; refusing an incorrect phrase mapping.")

    mapped = {}
    for span in phrases:
        start, end = prompt_start + span["start"], prompt_start + span["end"]
        selected = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not selected:
            raise ValueError(f"Phrase {span['phrase']!r} does not contain a Qwen token.")
        for i in selected:
            a, b = offsets[i]
            if a < start and full_text[a:start].strip():
                raise ValueError(f"Phrase {span['phrase']!r} starts inside a Qwen token; include the complete token.")
            if b > end and full_text[end:b].strip():
                raise ValueError(f"Phrase {span['phrase']!r} ends inside a Qwen token; include the complete token.")
            if i in mapped:
                raise ValueError("Weighted phrases overlap in a shared Qwen token.")
            if isinstance(pairs[i][0], dict):
                raise ValueError("A phrase unexpectedly mapped to a visual embedding.")
            mapped[i] = span["weight"]
        span["raw_token_indices"] = selected
    return mapped


def remap_phrase_rows(native_tokens, raw_weights, expanded_spans, priorities, prefix_length, final_length):
    pairs = native_tokens[_TOKEN_KEY][0]
    visual_locations = [(i, pair[0]) for i, pair in enumerate(pairs) if isinstance(pair[0], dict)]
    if len(visual_locations) != len(expanded_spans):
        raise ValueError("Visual placeholder count differs from Qwen's expanded image spans.")
    raw_to_expanded = {}
    cursor = 0
    span_index = 0
    for raw_index, pair in enumerate(pairs):
        if isinstance(pair[0], dict):
            if pair[0].get("type") != "image":
                raise ValueError("Unexpected non-image embedding; phrase weights cannot be aligned.")
            start, size, _grid = expanded_spans[span_index]
            if cursor != start or size < 1:
                raise ValueError("Expanded image span is inconsistent with native token sequence.")
            cursor += size
            span_index += 1
        else:
            raw_to_expanded[raw_index] = cursor
            cursor += 1
    if cursor != priorities.numel():
        raise ValueError("Qwen expanded token length differs from its captured conditioning rows.")
    keep = priorities > 0
    keep[:prefix_length] = False
    retained_indices = keep.long().cumsum(dim=0) - 1
    if int(keep.sum()) != final_length:
        raise ValueError("The encoded conditioning rows differ from the phrase mapping.")
    final_weights = [1.0] * final_length
    for raw_index, weight in raw_weights.items():
        expanded_index = raw_to_expanded.get(raw_index)
        if expanded_index is None or not bool(keep[expanded_index]):
            raise ValueError("A weighted phrase token was removed by reference conditioning.")
        result_index = int(retained_indices[expanded_index])
        final_weights[result_index] = weight
    return tuple(final_weights)
