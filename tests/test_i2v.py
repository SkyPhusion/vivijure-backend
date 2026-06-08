"""i2v's pure surface, tested on CPU: the frame-count math the temporal VAE constrains, the
duration it realizes, and the tier->profile decision. The Wan generation body needs torch and is
validated on a pod."""
from vivijure_backend.contract import Scene
from vivijure_backend.i2v import (
    DEFAULT_FPS,
    MAX_FRAMES,
    I2VParams,
    _step_callback,
    clip_seconds,
    frames_for,
    params_for,
    snap_frames,
)
from vivijure_backend.routing import QualityTier


# ----------------------------------------------------- per-step progress callback (item M)

def test_step_callback_reports_one_based_step_and_passes_kwargs_through():
    seen = []
    cb = _step_callback(lambda step, total: seen.append((step, total)), 40)
    kw = {"latents": "opaque"}
    out = cb(None, 11, 0, kw)          # diffusers signature: (pipe, step_index, timestep, kwargs)
    assert seen == [(12, 40)]          # reported 1-based: step_index 11 -> step 12 of 40
    assert out is kw                   # callback_kwargs returned unchanged, loop unaffected


def test_step_callback_is_none_without_a_progress_cb():
    # No cb -> None, so animate omits callback_on_step_end entirely (zero overhead).
    assert _step_callback(None, 40) is None


def test_step_callback_swallows_a_failing_progress_cb():
    def boom(step, total):
        raise RuntimeError("progress write failed")
    cb = _step_callback(boom, 4)
    kw = {"latents": 1}
    assert cb(None, 0, 0, kw) is kw    # must not raise; render continues


def _scene(target=None):
    d = {"id": "s", "prompt": "x"}
    if target is not None:
        d["target_seconds"] = target
    return Scene.from_dict(d, 0)


# --------------------------------------------------------------------- frame math (4k+1)

def test_snap_frames_to_4k_plus_1():
    assert snap_frames(81) == 81   # already valid
    assert snap_frames(80) == 81   # round up to the next valid count
    assert snap_frames(5) == 5
    assert snap_frames(6) == 9
    assert snap_frames(4) == 5
    assert snap_frames(1) == 1


def test_frames_for_target_duration():
    assert frames_for(5, 16) == 81       # 80 -> snapped 81
    assert frames_for(4, 16) == 65       # 64 -> snapped 65
    assert frames_for(10, 16) == MAX_FRAMES  # capped at the model ceiling
    assert frames_for(None) == MAX_FRAMES    # no target -> ceiling
    assert frames_for(0) == MAX_FRAMES


def test_clip_seconds_is_frames_over_fps():
    # 81/16 = 5.0625; Python rounds half-to-even at 3 places -> 5.062
    assert clip_seconds(81, 16) == 5.062
    assert clip_seconds(65, 16) == 4.062


# ----------------------------------------------------------------------- tier profiles

def test_final_tier_is_full_step():
    p = params_for(_scene(5), QualityTier.FINAL)
    assert p.distill is False
    assert p.steps == 40
    assert p.guidance_scale == 5.0
    assert p.num_frames == 81


def test_draft_and_standard_are_distilled():
    for tier in (QualityTier.DRAFT, QualityTier.STANDARD):
        p = params_for(_scene(4), tier)
        assert p.distill is True
        assert p.steps == 4
        assert p.guidance_scale == 1.0
        assert p.num_frames == 65   # 4s at 16fps, snapped


def test_params_for_uses_scene_target_for_frame_count():
    assert params_for(_scene(2), QualityTier.DRAFT).num_frames == snap_frames(2 * DEFAULT_FPS)
    assert params_for(_scene(None), QualityTier.DRAFT).num_frames == MAX_FRAMES


def test_default_params_are_the_distilled_path():
    p = I2VParams()
    assert p.distill is True
    assert p.steps == 4
    assert p.fps == DEFAULT_FPS
    assert p.num_frames == MAX_FRAMES
