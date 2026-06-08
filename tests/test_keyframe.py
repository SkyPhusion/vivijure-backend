"""The keyframe engine's decisions are pure and tested on CPU: the prompt it builds, the region
geometry that confines each identity, and the single-vs-regional path choice. The generation
body needs torch/diffusers/PIL and is validated on a pod."""
from vivijure_backend.contract import Cast, Scene, Storyboard
from vivijure_backend.keyframe import (
    DEFAULT_IP_ADAPTER_SCALE,
    DEFAULT_LORA_SCALE,
    KeyframeParams,
    build_prompt,
    engine_for,
    region_boxes,
    slot_trigger,
)

CAST = Cast.from_registry({"characters": {
    "A": {"name": "Vesper", "prompt": "teal-haired netrunner"},
    "B": {"name": "Rhode", "prompt": "bearded fixer"},
}})


def _sb(**extra):
    return Storyboard.from_dict({"style_prefix": "cyberpunk anime,", "use_characters": ["A", "B"],
                                 "scenes": [{"id": "s", "prompt": "x"}], **extra})


# --------------------------------------------------------------------------- triggers/prompt

def test_slot_trigger_is_the_character_name():
    assert slot_trigger(CAST, "A") == "Vesper"
    assert slot_trigger(CAST, "Z") == "Z"  # unknown slot falls back to the slot id


def test_prompt_combines_style_scene_and_triggers():
    sc = Scene.from_dict({"id": "shot_01", "prompt": "rooftop standoff", "character_slots": ["A", "B"]}, 0)
    p = build_prompt(sc, CAST, _sb())
    assert p == "cyberpunk anime, rooftop standoff, Vesper, Rhode"


def test_prompt_has_no_dangling_comma_without_style():
    sc = Scene.from_dict({"id": "s", "prompt": "wide skyline", "character_slots": []}, 0)
    p = build_prompt(sc, CAST, _sb(style_prefix=""))
    assert p == "wide skyline"  # empty style + no characters -> no leading/trailing comma


def test_prompt_appends_real_style_preset():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A"]}, 0)
    p = build_prompt(sc, CAST, _sb(style_preset="neon noir"))
    assert p.endswith("neon noir")
    # the literal "None" preset is never appended
    assert "None" not in build_prompt(sc, CAST, _sb())


# --------------------------------------------------------------------------- region geometry

def test_single_region_is_full_canvas():
    assert region_boxes(1024, 1024, 1) == [(0, 0, 1024, 1024)]


def test_two_regions_split_vertically():
    boxes = region_boxes(1024, 1024, 2)
    assert boxes == [(0, 0, 512, 1024), (512, 0, 1024, 1024)]


def test_last_region_absorbs_odd_remainder():
    # 1025 / 2 = 512 step; the last region must reach the full width, not stop at 1024
    boxes = region_boxes(1025, 1024, 2)
    assert boxes[-1][2] == 1025


def test_horizontal_orientation_stacks():
    boxes = region_boxes(512, 1024, 2, orientation="horizontal")
    assert boxes == [(0, 0, 512, 512), (0, 512, 512, 1024)]


# ----------------------------------------------------------------------- engine path choice

def test_single_path_for_zero_or_one_character():
    assert engine_for(Scene.from_dict({"id": "s", "prompt": "x", "character_slots": []}, 0), KeyframeParams()) == "single"
    assert engine_for(Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A"]}, 0), KeyframeParams()) == "single"


def test_regional_path_for_two_characters():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A", "B"]}, 0)
    assert engine_for(sc, KeyframeParams()) == "regional"


def test_over_cap_falls_back_to_single():
    sc = Scene.from_dict({"id": "s", "prompt": "x", "character_slots": ["A", "B", "C"]}, 0)
    assert engine_for(sc, KeyframeParams(max_slots=2)) == "single"


# --------------------------------------------------------------------------- params/defaults

def test_anti_bleed_defaults():
    p = KeyframeParams()
    assert p.lora_scale == DEFAULT_LORA_SCALE == 0.3
    assert p.ip_adapter_scale == DEFAULT_IP_ADAPTER_SCALE == 0.7
    assert p.pose_conditioning is True
    assert p.max_slots == 2
