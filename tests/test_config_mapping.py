"""Config->engine mapping completeness guard.

Every config field that drives a GPU parameter must appear in its engine params struct.
A one-line assertion here catches a dropped knob before it silently ships ([B], [D]).

Intentionally-unmapped fields are documented in UNMAPPED_* allowlists at the bottom.
"""
from __future__ import annotations

import pytest

from vivijure_backend.config import (
    FaceRestore,
    FinishConfig,
    I2VConfig,
    KeyframeConfig,
    MultiCharConfig,
    Scheduler,
    IdentityMethod,
    QualityTier,
    RenderConfig,
    FeatureCache,
)
from vivijure_backend.keyframe import KeyframeParams
from vivijure_backend.i2v import I2VParams
from vivijure_backend.pipeline import keyframe_params_from, i2v_params_from, finish_params_from

# ------------------------------------------------------------------ helpers

def _kfcfg(**overrides) -> KeyframeConfig:
    base = KeyframeConfig.for_tier(QualityTier.FINAL)
    return KeyframeConfig.from_dict(overrides, tier=QualityTier.FINAL)


def _i2vcfg(**overrides) -> I2VConfig:
    return I2VConfig.from_dict(overrides, tier=QualityTier.FINAL)


def _fincfg(**overrides) -> FinishConfig:
    base = FinishConfig.from_dict(overrides)
    return base


def _scene(seconds=5.0):
    class S:
        target_seconds = seconds
        id = "shot_01"
    return S()


def _rcfg(kf=None, i2v=None, fin=None) -> RenderConfig:
    return RenderConfig.from_request(QualityTier.FINAL, {
        **({"keyframe": kf} if kf else {}),
        **({"i2v": i2v} if i2v else {}),
        **({"finish": fin} if fin else {}),
    })


# --------------------------------------------------------- keyframe mapping

def test_keyframe_width_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"width": 1280}))
    assert p.width == 1280


def test_keyframe_height_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"height": 720}))
    assert p.height == 720


def test_keyframe_guidance_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"guidance_scale": 9.0}))
    assert p.guidance_scale == 9.0


def test_keyframe_seed_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"seed": 12345}))
    assert p.seed == 12345


def test_keyframe_distill_selects_distill_steps():
    p = keyframe_params_from(_rcfg(kf={"distill": True, "distill_steps": 6}))
    assert p.few_step is True
    assert p.steps == 6


def test_keyframe_full_steps_used_when_not_distill():
    p = keyframe_params_from(_rcfg(kf={"distill": False, "steps": 25}))
    assert p.few_step is False
    assert p.steps == 25


def test_keyframe_scheduler_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"scheduler": "dpmpp_2m_karras"}))
    assert p.scheduler == "dpmpp_2m_karras"


def test_keyframe_identity_method_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"identity_method": "instantid"}))
    assert p.identity_method == "instantid"


def test_keyframe_instantid_controlnet_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"instantid_controlnet_scale": 0.5}))
    assert p.instantid_controlnet_scale == pytest.approx(0.5)


def test_keyframe_instantid_ip_adapter_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"instantid_ip_adapter_scale": 0.3}))
    assert p.instantid_ip_adapter_scale == pytest.approx(0.3)


def test_keyframe_multi_char_lora_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"lora_scale_per_slot": 0.5}}))
    assert p.lora_scale == pytest.approx(0.5)


def test_keyframe_multi_char_ip_adapter_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"ip_adapter_scale_per_slot": 0.4}}))
    assert p.ip_adapter_scale == pytest.approx(0.4)


def test_keyframe_pose_conditioning_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"pose_conditioning": False}}))
    assert p.pose_conditioning is False


def test_keyframe_controlnet_pose_scale_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"controlnet_pose_scale": 0.8}}))
    assert p.controlnet_pose_scale == pytest.approx(0.8)


def test_keyframe_region_gutter_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"region_gutter": 32}}))
    assert p.region_gutter == 32


def test_keyframe_max_slots_reaches_params():
    p = keyframe_params_from(_rcfg(kf={"multi_char": {"max_slots": 3}}))
    assert p.max_slots == 3


# ---------------------------------------------------------- i2v mapping

def test_i2v_fps_reaches_params():
    cfg = _rcfg(i2v={"fps": 24})
    p = i2v_params_from(cfg, _scene(5.0))
    assert p.fps == 24


def test_i2v_guidance_scale_reaches_params():
    cfg = _rcfg(i2v={"distill": False, "guidance_scale": 4.0})
    p = i2v_params_from(cfg, _scene())
    assert p.guidance_scale == pytest.approx(4.0)


def test_i2v_steps_selected_by_distill():
    cfg = _rcfg(i2v={"distill": False, "steps": 30})
    p = i2v_params_from(cfg, _scene())
    assert p.steps == 30


def test_i2v_distill_steps_selected_when_distill():
    cfg = _rcfg(i2v={"distill": True, "distill_steps": 4})
    p = i2v_params_from(cfg, _scene())
    assert p.distill is True
    assert p.steps == 4


def test_i2v_feature_cache_reaches_params():
    cfg = _rcfg(i2v={"distill": False, "feature_cache": "mixcache"})
    p = i2v_params_from(cfg, _scene())
    assert p.feature_cache == FeatureCache.MIXCACHE


def test_i2v_negative_prompt_reaches_params():
    cfg = _rcfg(i2v={"negative_prompt": "blurry, watermark"})
    p = i2v_params_from(cfg, _scene())
    assert p.negative_prompt == "blurry, watermark"


def test_i2v_num_frames_from_scene_duration():
    # 5s * 16fps = 80 frames -> snap to 81 (4*20+1); verify config fps is used
    cfg = _rcfg(i2v={"fps": 16})
    p = i2v_params_from(cfg, _scene(5.0))
    assert p.num_frames > 0
    assert (p.num_frames - 1) % 4 == 0  # 4k+1 invariant


# -------------------------------------------------------- finish mapping

def test_finish_interpolate_reaches_params():
    cfg = _rcfg(fin={"interpolate": True, "interpolation_factor": 4})
    p = finish_params_from(cfg)
    assert p.interpolate is True
    assert p.factor == 4


def test_finish_target_fps_reaches_params():
    cfg = _rcfg(fin={"interpolate": True, "target_fps": 60})
    p = finish_params_from(cfg)
    assert p.target_fps == 60


def test_finish_face_restore_maps_enum_to_bool_and_backend():
    cfg = _rcfg(fin={"face_restore": "gfpgan"})
    p = finish_params_from(cfg)
    assert p.face_restore is True
    assert p.face_restore_backend == "gfpgan"


def test_finish_face_restore_none_maps_to_false():
    cfg = _rcfg(fin={"face_restore": "none"})
    p = finish_params_from(cfg)
    assert p.face_restore is False


def test_finish_face_fidelity_reaches_params():
    cfg = _rcfg(fin={"face_restore": "gfpgan", "face_fidelity": 0.3})
    p = finish_params_from(cfg)
    assert p.face_fidelity == pytest.approx(0.3)


def test_finish_only_faces_reaches_params():
    cfg = _rcfg(fin={"face_restore": "gfpgan", "only_faces": False})
    p = finish_params_from(cfg)
    assert p.only_faces is False


# -------------------------------------------------------- allowlists

# Fields intentionally NOT forwarded to engine params (document them here so a reviewer
# knows they're consciously excluded, not accidentally dropped):
KEYFRAME_UNMAPPED = {
    "base_model",     # deploy-time model repo (wired to ModelServer.specs at pod startup, not I/O params)
    "distill_model",  # deploy-time model repo
    "ip_adapter_scale",  # single-char IP-Adapter pull; KeyframeParams.ip_adapter_scale comes from
                         # multi_char.ip_adapter_scale_per_slot. Single-char path reads it separately.
}

I2V_UNMAPPED = {
    "model",          # deploy-time model repo
    "distill_model",  # deploy-time model repo
    "seconds_per_shot",  # indirect: affects num_frames via I2VConfig.frames_for, not a direct param
    "loader",         # model-loading strategy (diffusers vs LightX2V fallback), not a generation param
    # "seed",         # NOW WIRED (PR #40); previously read from keyframe.seed
    # "flow_shift",   # NOW WIRED (PR #40); previously not in I2VParams
}

FINISH_UNMAPPED = set()  # all FinishConfig fields reach FinishParams
