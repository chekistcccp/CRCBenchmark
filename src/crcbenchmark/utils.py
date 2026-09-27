from __future__ import annotations
import hashlib
import json
import math
import random
import re
from pathlib import Path
import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_first_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    starts = [i for i, c in enumerate(text) if c in "[{"]
    for s in starts:
        opener = text[s]
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_str = False
        esc = False
        for i in range(s, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[s:i + 1])
                    except Exception:
                        break
    return None


def clean_model_answer(text: str) -> str:
    """Remove only known transport terminators and an optional JSON fence."""
    answer = str(text).strip()
    answer = re.sub(r"(?:<turn\|>|<\|im_end\|>|</s>|<eos>)+\s*$", "", answer).strip()
    match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", answer, re.IGNORECASE)
    return match.group(1).strip() if match else answer


def _single_marked_payload(text: str) -> str | None:
    """Read one explicit GLM answer span without guessing from surrounding prose."""
    blocks = re.findall(r"<\|begin_of_box\|>([\s\S]*?)<\|end_of_box\|>", str(text))
    if len(blocks) != 1 or str(text).count("<|begin_of_box|>") != 1 or str(text).count("<|end_of_box|>") != 1:
        return None
    return blocks[0].strip()


def parse_choice_response(text: str, choices: list[str]) -> str | None:
    answer = clean_model_answer(text).upper()
    allowed = {str(choice).upper(): choice for choice in choices}
    if answer in allowed:
        return allowed[answer]
    marked = _single_marked_payload(text)
    return allowed.get(marked.upper()) if marked is not None else None


def parse_label_list_response(text: str, allowed: str, max_items: int | None = None) -> list[str] | None:
    try:
        parsed = json.loads(clean_model_answer(text))
    except (TypeError, ValueError):
        # A single fenced final list is still an unambiguous structured answer,
        # even when the model surrounds it with an explanation.
        blocks = re.findall(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", str(text), re.IGNORECASE)
        if len(blocks) != 1:
            marked = _single_marked_payload(text)
            blocks = [marked] if marked is not None else []
        if len(blocks) != 1:
            return None
        try:
            parsed = json.loads(blocks[0])
        except (TypeError, ValueError):
            return None
    if not isinstance(parsed, list) or not parsed or (max_items is not None and len(parsed) > max_items):
        return None
    labels = [x.upper() for x in parsed if isinstance(x, str)]
    if len(labels) != len(parsed) or len(labels) != len(set(labels)):
        return None
    if labels == ["NONE"]:
        return labels
    if any(len(x) != 1 or x not in allowed for x in labels):
        return None
    return labels


def parse_json_object_response(text: str) -> dict | None:
    try:
        parsed = json.loads(clean_model_answer(text))
    except (TypeError, ValueError):
        blocks = re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", str(text), re.IGNORECASE)
        if len(blocks) != 1:
            marked = _single_marked_payload(text)
            blocks = [marked] if marked is not None else []
        if len(blocks) != 1:
            return None
        try:
            parsed = json.loads(blocks[0])
        except (TypeError, ValueError):
            return None
    return parsed if isinstance(parsed, dict) else None


def parse_t2_response(text: str) -> dict | None:
    """Require bounded coordinates and a box with positive area."""
    answer = parse_json_object_response(text)
    if answer is None:
        return None
    point, box = answer.get("point"), answer.get("box")
    if not (isinstance(point, list) and len(point) == 2 and isinstance(box, list) and len(box) == 4):
        return None
    if not all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1000 for x in point + box):
        return None
    if not (box[0] < box[2] and box[1] < box[3]):
        return None
    return answer


def natural_key(value: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", value)]
