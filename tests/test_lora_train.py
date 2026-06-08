"""CPU-testable surface of the LoRA trainer: the parts that decide *what* to train before
any GPU is touched. The training loop itself needs CUDA and is validated on the pod."""
from pathlib import Path

import pytest

from vivijure_backend.contract import Character
from vivijure_backend.lora_train import (
    LoraTrainConfig,
    TrainedLora,
    caption_for,
    default_base_repo,
    train_slot,
)


def _char(slot="A", name="Vesper", prompt="teal-haired netrunner", refs=()):
    return Character(slot=slot, name=name, prompt=prompt, ref_paths=[Path(p) for p in refs])


def test_default_base_repo_is_the_keyframe_sdxl():
    # The LoRA must train against the same checkpoint the keyframe stage draws with.
    assert default_base_repo() == "SG161222/RealVisXL_V5.0"


def test_caption_uses_name_as_trigger_and_appends_prompt():
    assert caption_for(_char(), LoraTrainConfig().caption_template) == "Vesper, teal-haired netrunner"


def test_caption_falls_back_to_slot_when_unnamed():
    assert caption_for(_char(name="", prompt=""), LoraTrainConfig().caption_template) == "A"


def test_caption_drops_dangling_comma_when_prompt_is_empty():
    # An empty prompt must not leak a trailing ", " into the caption.
    assert caption_for(_char(prompt=""), LoraTrainConfig().caption_template) == "Vesper"


def test_train_slot_rejects_a_character_with_no_refs():
    # Refuse on the CPU before allocating a GPU for a slot that has nothing to learn from.
    with pytest.raises(ValueError, match="no reference images"):
        train_slot(_char(refs=()), Path("/tmp/never"))


def test_config_defaults_fit_a_few_reference_character():
    cfg = LoraTrainConfig()
    assert cfg.rank == 16
    assert cfg.resolution == 1024
    assert cfg.gradient_checkpointing is True
    assert cfg.max_steps == 1000


def test_trained_lora_carries_the_trigger_token():
    tl = TrainedLora(slot="A", path=Path("a.safetensors"), trigger="Vesper",
                     steps=1000, rank=16, ref_count=8, base_repo="x")
    assert tl.trigger == "Vesper"
