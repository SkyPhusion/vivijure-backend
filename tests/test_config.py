"""CPU tests for the typed generation config. These assert the contract both sides build to:
the tier baselines, forgiving + clamped parsing, and the invalid-combination guards. No GPU."""
from vivijure_backend.config import (
    FeatureCache,
    I2VConfig,
    I2VLoader,
    IdentityMethod,
    KeyframeConfig,
    MultiCharConfig,
    Scheduler,
)
from vivijure_backend.models import DEFAULT_SPECS, ModelRole
from vivijure_backend.orchestrator import MULTI_CHAR_DEFAULTS
from vivijure_backend.routing import QualityTier


# ------------------------------------------------------------------- defaults / cross-refs

def test_keyframe_base_defaults_to_the_spec_sdxl():
    assert KeyframeConfig().base_model == DEFAULT_SPECS[ModelRole.KEYFRAME_BASE].repo_id


def test_i2v_model_defaults_to_the_spec_wan():
    assert I2VConfig().model == DEFAULT_SPECS[ModelRole.I2V].repo_id
    assert I2VConfig().distill_model == DEFAULT_SPECS[ModelRole.I2V_DISTILL].repo_id


def test_multichar_defaults_mirror_orchestrator():
    # The anti-bleed scales must agree with what the planner already records.
    mc = MultiCharConfig()
    assert mc.regional is (MULTI_CHAR_DEFAULTS["engine"] == "regional")
    assert mc.pose_conditioning == MULTI_CHAR_DEFAULTS["pose_conditioning"]
    assert mc.lora_scale_per_slot == MULTI_CHAR_DEFAULTS["lora_scale_per_slot"]
    assert mc.ip_adapter_scale_per_slot == MULTI_CHAR_DEFAULTS["ip_adapter_scale_per_slot"]
    assert mc.max_slots == MULTI_CHAR_DEFAULTS["max_slots"]


# --------------------------------------------------------------------------- tier baselines

def test_keyframe_draft_is_four_step_distilled_cfg_zero():
    k = KeyframeConfig.for_tier(QualityTier.DRAFT)
    assert k.distill and k.distill_steps == 4 and k.steps == 4
    assert k.guidance_scale == 0.0
    assert k.scheduler is Scheduler.DDIM_TRAILING


def test_keyframe_final_is_full_step_high_cfg_no_distill():
    k = KeyframeConfig.for_tier(QualityTier.FINAL)
    assert not k.distill and k.steps == 30 and k.guidance_scale > 0


def test_i2v_draft_is_lightning_four_step_no_cache():
    v = I2VConfig.for_tier(QualityTier.DRAFT)
    assert v.distill and v.distill_steps == 4 and v.steps == 4
    assert v.guidance_scale == 1.0
    assert v.feature_cache is FeatureCache.NONE  # never cache a 4-step render


def test_i2v_final_is_full_step_with_cache():
    v = I2VConfig.for_tier(QualityTier.FINAL)
    assert not v.distill and v.steps == 40
    assert v.feature_cache is FeatureCache.MIXCACHE


# --------------------------------------------------------------------------- forgiving parse

def test_from_dict_ignores_unknown_keys():
    k = KeyframeConfig.from_dict({"totally_made_up": 7, "steps": 12})
    assert k.steps == 12


def test_from_dict_clamps_out_of_range():
    k = KeyframeConfig.from_dict({"steps": 9999, "guidance_scale": -4})
    assert k.steps == 128       # clamped to the documented ceiling
    assert k.guidance_scale == 0.0
    v = I2VConfig.from_dict({"num_frames": 100000, "fps": 0})
    assert v.num_frames == 256
    assert v.fps == 1


def test_from_dict_layers_over_tier_baseline():
    # Tier sets the baseline; the dict overrides only what it names.
    k = KeyframeConfig.from_dict({"steps": 6}, tier=QualityTier.FINAL)
    assert k.steps == 6
    assert k.distill is False           # inherited from the FINAL baseline
    assert k.guidance_scale == 6.5      # inherited


def test_resolution_string_is_parsed():
    k = KeyframeConfig.from_dict({"resolution": "1344x768"})
    assert (k.width, k.height) == (1344, 768)


def test_bad_resolution_string_falls_back():
    k = KeyframeConfig.from_dict({"resolution": "wide"})
    assert (k.width, k.height) == (1024, 1024)


# ------------------------------------------------------------------- invalid-combo guards

def test_distill_forces_cache_off_even_when_overridden():
    # Caching a 4-step distilled render is invalid; an override cannot create it.
    v = I2VConfig.from_dict({"distill": True, "feature_cache": "mixcache"})
    assert v.distill is True
    assert v.feature_cache is FeatureCache.NONE


def test_enum_parse_is_forgiving():
    v = I2VConfig.from_dict({"loader": "nonsense"})
    assert v.loader is I2VLoader.DIFFUSERS
    k = KeyframeConfig.from_dict({"identity_method": "huh"})
    assert k.identity_method is IdentityMethod.IP_ADAPTER


def test_identity_method_is_keyframe_level_default_ip_adapter():
    # Identity applies to single AND multi-char shots, so it lives on KeyframeConfig, not
    # multi_char. IP-Adapter is the default everywhere; InstantID is a single-char upgrade.
    assert KeyframeConfig().identity_method is IdentityMethod.IP_ADAPTER
    k = KeyframeConfig.from_dict({"identity_method": "instantid", "instantid_controlnet_scale": 0.9})
    assert k.identity_method is IdentityMethod.INSTANTID
    assert k.instantid_controlnet_scale == 0.9
    # The regional block carries no identity method (masked IP-Adapter only, no InstantID).
    assert not hasattr(MultiCharConfig(), "identity_method")


# --------------------------------------------------------------------------- derivations

def test_frames_for_derives_from_seconds_and_fps():
    v = I2VConfig(fps=16)
    assert v.frames_for(5.0) == 80
    assert v.frames_for(None) == v.num_frames    # no duration -> the configured default
    assert v.frames_for(10_000) == 256           # clamped to ceiling


def test_round_trips_through_to_dict():
    k = KeyframeConfig.for_tier(QualityTier.STANDARD)
    rk = KeyframeConfig.from_dict(k.to_dict())
    assert rk.steps == k.steps
    assert rk.scheduler is k.scheduler                      # enum survives the round-trip
    assert rk.identity_method is k.identity_method
    assert rk.multi_char.lora_scale_per_slot == k.multi_char.lora_scale_per_slot
    v = I2VConfig.for_tier(QualityTier.FINAL)
    rv = I2VConfig.from_dict(v.to_dict())
    assert rv.feature_cache is v.feature_cache and rv.loader is v.loader
