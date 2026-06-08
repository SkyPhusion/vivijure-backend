"""The quant matrix is the load-bearing claim: SDXL is a UNet and never reaches NVFP4 even
on a fp4-capable card, video stays fp8, and adapters load at base dtype. Asserted on CPU."""
from vivijure_backend.device import Device, Quant
from vivijure_backend.models import ModelFamily, ModelRole, ModelServer, quant_for

B200 = Device.classify((10, 0), "NVIDIA B200")
H200 = Device.classify((9, 0), "NVIDIA H200")
CPU = Device.classify((0, 0), "cpu")


def test_sdxl_never_reaches_fp4_even_on_blackwell():
    assert quant_for(ModelFamily.SDXL_UNET, B200) is Quant.FP8
    assert quant_for(ModelFamily.SDXL_UNET, H200) is Quant.FP8


def test_dit_gets_fp4_only_where_the_card_supports_it():
    assert quant_for(ModelFamily.DIT, B200) is Quant.NVFP4
    assert quant_for(ModelFamily.DIT, H200) is Quant.FP8  # Hopper has no fp4 engine


def test_video_dit_is_fp8_on_both_archs():
    assert quant_for(ModelFamily.VIDEO_DIT, B200) is Quant.FP8
    assert quant_for(ModelFamily.VIDEO_DIT, H200) is Quant.FP8


def test_aux_always_loads_at_base_dtype():
    for dev in (B200, H200, CPU):
        assert quant_for(ModelFamily.AUX, dev) is Quant.BF16


def test_everything_falls_to_bf16_without_fp8():
    for fam in ModelFamily:
        assert quant_for(fam, CPU) is Quant.BF16


def test_only_dit_is_fp4_capable():
    assert ModelFamily.DIT.fp4_capable
    assert not ModelFamily.SDXL_UNET.fp4_capable
    assert not ModelFamily.VIDEO_DIT.fp4_capable


def test_model_server_plan_needs_no_gpu():
    plan = ModelServer(device=B200).plan()
    assert plan[ModelRole.KEYFRAME_BASE.value] == "fp8"   # SDXL
    assert plan[ModelRole.I2V.value] == "fp8"             # Wan video DiT
    assert plan[ModelRole.CONTROLNET_POSE.value] == "bf16"  # aux
    # every default role is represented
    assert set(plan) == {r.value for r in ModelRole}


def test_default_specs_cover_every_role():
    from vivijure_backend.models import DEFAULT_SPECS
    assert set(DEFAULT_SPECS) == set(ModelRole)
    for role, spec in DEFAULT_SPECS.items():
        assert spec.role is role
        assert spec.repo_id  # a real, non-empty HF id
