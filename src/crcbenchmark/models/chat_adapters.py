from __future__ import annotations

import math

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from .base import VLMAdapter
from .pipeline_adapter import _dtype


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _install_remote_code_transformers_compat():
    """
    Backfill tied-weight bookkeeping expected by newer Transformers loaders.

    InternVL/MiniCPM remote-code model classes were authored against the
    Transformers 4.x API and may only expose _tied_weights_keys.  The frozen
    benchmark runtime is Transformers 4.57.6, but this fallback also prevents
    accidental 5.x environments from failing with all_tied_weights_keys.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
    except Exception:
        pass


def _internvl_transform(input_size=448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(
    image,
    min_num=1,
    max_num=12,
    image_size=448,
    use_thumbnail=True,
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    ratio = _closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * ratio[0]
    target_height = image_size * ratio[1]
    blocks = ratio[0] * ratio[1]

    resized = image.resize((target_width, target_height))
    processed = []

    for i in range(blocks):
        box = (
            (i % ratio[0]) * image_size,
            (i // ratio[0]) * image_size,
            ((i % ratio[0]) + 1) * image_size,
            ((i // ratio[0]) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def _load_internvl_image(path, dtype, input_size=448, max_num=12):
    image = Image.open(path).convert("RGB")
    transform = _internvl_transform(input_size)
    tiles = _internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = torch.stack([transform(x) for x in tiles])
    return pixel_values.to(dtype=dtype, device="cuda")


class InternVLChatVLM(VLMAdapter):
    """Official-style InternVL remote-code AutoModel + model.chat adapter."""

    def __init__(
        self,
        model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_new_tokens=96,
    ):
        self.model_path = str(model_path)
        self.dtype = _dtype(dtype)
        self.max_new_tokens = int(max_new_tokens)

        _install_remote_code_transformers_compat()
        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            use_fast=False,
            local_files_only=True,
        )

    def generate(self, image_path, prompt, max_new_tokens=None):
        pixel_values = _load_internvl_image(
            image_path,
            dtype=self.dtype,
            input_size=448,
            max_num=12,
        )
        question = "<image>\n" + prompt
        generation_config = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": False,
        }
        with torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )
        return str(response).strip()

    def score_choices(self, image_path, prompt, choices):
        return None


class MiniCPMChatVLM(VLMAdapter):
    """Official-style MiniCPM-V-4.5 remote-code AutoModel + model.chat adapter."""

    def __init__(
        self,
        model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        max_new_tokens=96,
    ):
        self.model_path = str(model_path)
        self.dtype = _dtype(dtype)
        self.max_new_tokens = int(max_new_tokens)

        _install_remote_code_transformers_compat()
        self.model = (
            AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=trust_remote_code,
                attn_implementation="sdpa",
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )
            .eval()
            .cuda()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )

    def generate(self, image_path, prompt, max_new_tokens=None):
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [image, prompt]}]

        with torch.inference_mode():
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=False,
                enable_thinking=False,
                max_new_tokens=int(max_new_tokens or self.max_new_tokens),
            )
        return str(response).strip()

    def score_choices(self, image_path, prompt, choices):
        return None
