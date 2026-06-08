from vivijure_backend.device import Tier
from vivijure_backend.routing import QualityTier, Stage, gpu_for


def test_i2v_climbs_tiers_with_quality():
    assert gpu_for(Stage.I2V, QualityTier.DRAFT) is Tier.RTX_PRO_6000
    assert gpu_for(Stage.I2V, QualityTier.STANDARD) is Tier.H200
    assert gpu_for(Stage.I2V, QualityTier.FINAL) is Tier.B200


def test_cheap_stages_stay_on_entry_card():
    for q in QualityTier:
        assert gpu_for(Stage.LORA_TRAIN, q) is Tier.RTX_PRO_6000
        assert gpu_for(Stage.KEYFRAME, q) is Tier.RTX_PRO_6000


def test_assemble_is_off_gpu():
    for q in QualityTier:
        assert gpu_for(Stage.ASSEMBLE, q) is None


def test_quality_tier_parse_is_forgiving():
    assert QualityTier.parse("DRAFT") is QualityTier.DRAFT
    assert QualityTier.parse("standard") is QualityTier.STANDARD
    assert QualityTier.parse(None) is QualityTier.FINAL       # default
    assert QualityTier.parse("nonsense") is QualityTier.FINAL  # unknown -> final
