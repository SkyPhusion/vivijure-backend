"""Image-to-video: animate each keyframe into its shot's clip.

The keyframe is the still; this turns it into motion. Wan 2.2 image-to-video takes the keyframe
as the first frame and the scene prompt as the motion description and produces N frames at a
target fps. This is the long pole of the whole render, so the speed knobs are the point: the
draft and standard tiers run a few-step distilled path (the Wan2.2-Lightning LoRA, ~4 steps) for
the big throughput win, while the final tier runs full steps for the hero clip. The planner
already decided which shots animate and on which GPU tier; this just executes one shot.

Clean-room: built from diffusers' WanImageToVideoPipeline + export_to_video, the Wan2.2-Lightning
distill card, and the LightX2V fallback loader (diffusers LoRA-load issue #12535), not any prior
pipeline. The frame-count / duration math and the tier->steps decision are pure and CPU-tested;
the generation body defers torch/diffusers and is validated on a pod. Engine knobs live in
`I2VParams`; the control plane's typed `I2VConfig` (separate work) maps into them.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .contract import Scene
from .routing import QualityTier

# Wan's temporal VAE compresses time by 4, so a clip's frame count must be 4k+1 (e.g. 81 frames
# = 4*20+1, ~5s at 16 fps). FPS and the frame ceiling follow the A14B i2v defaults.
TEMPORAL_STRIDE = 4
DEFAULT_FPS = 16
MAX_FRAMES = 81  # ~5s at 16 fps, the model's comfortable clip length


@dataclass
class I2VParams:
    """Engine knobs for one shot's animation. Defaults are the few-step distilled path (the
    throughput win); the final tier flips `distill` off for full steps. The control plane's
    I2VConfig fills these per job."""
    num_frames: int = MAX_FRAMES
    fps: int = DEFAULT_FPS
    steps: int = 4                   # 4-step Wan2.2-Lightning distill
    guidance_scale: float = 1.0      # distilled sampling runs (near-)guidance-free
    distill: bool = True
    seed: int = 0
    height: int | None = None        # default: follow the keyframe's size
    width: int | None = None
    negative_prompt: str = "static, still, frozen, jpeg artifacts, blurry, watermark"


@dataclass
class I2VResult:
    shot_id: str
    path: Path
    num_frames: int
    fps: int
    seconds: float
    distilled: bool


# --------------------------------------------------------------------------- pure helpers

def snap_frames(n: int) -> int:
    """Snap a frame count to the nearest valid 4k+1 the temporal VAE accepts (rounding up so a
    clip never comes out shorter than asked), clamped to at least one frame."""
    n = max(1, int(n))
    rem = (n - 1) % TEMPORAL_STRIDE
    return n if rem == 0 else n + (TEMPORAL_STRIDE - rem)


def frames_for(target_seconds: float | None, fps: int = DEFAULT_FPS, *, max_frames: int = MAX_FRAMES) -> int:
    """Frame count for a target duration at `fps`: snap to 4k+1 and cap at the model ceiling.
    Falls back to the ceiling when the scene gives no target."""
    if not target_seconds or target_seconds <= 0:
        return max_frames
    return min(max_frames, snap_frames(round(target_seconds * fps)))


def clip_seconds(num_frames: int, fps: int = DEFAULT_FPS) -> float:
    """The realized clip length. i2v fixes the first frame to the keyframe, so N frames play as
    N/fps seconds."""
    return round(num_frames / fps, 3)


def params_for(scene: Scene, quality: QualityTier, *, base: I2VParams | None = None) -> I2VParams:
    """Resolve the per-shot params: frame count from the scene's target duration, and the
    step/guidance/distill profile from the quality tier (draft/standard distilled for throughput,
    final full-step for the hero clip)."""
    p = base or I2VParams()
    p.num_frames = frames_for(scene.target_seconds, p.fps)
    if quality is QualityTier.FINAL:
        p.distill, p.steps, p.guidance_scale = False, 40, 5.0
    else:  # draft / standard: the few-step distilled path
        p.distill, p.steps, p.guidance_scale = True, 4, 1.0
    return p


# --------------------------------------------------------------------------- animate (GPU)

def animate(
    scene: Scene,
    keyframe: Path,
    prompt: str,
    server,
    out_path: Path,
    *,
    params: I2VParams | None = None,
) -> I2VResult:
    """Animate `keyframe` into a clip at `out_path` for one scene.

    `server` is a `models.ModelServer` (provides the Wan i2v pipeline with the Lightning distill
    LoRA). The keyframe is the first frame; `prompt` describes the motion. Heavy imports are
    deferred; the body is validated on a pod.
    """
    cfg = params or I2VParams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch  # deferred: keep this module CPU-importable
    from diffusers.utils import export_to_video, load_image

    image = load_image(str(keyframe))
    height = cfg.height or image.height
    width = cfg.width or image.width

    pipe = server.i2v_pipeline()
    _set_distill(pipe, cfg.distill)
    frames = pipe(
        image=image, prompt=prompt, negative_prompt=cfg.negative_prompt,
        height=height, width=width, num_frames=cfg.num_frames,
        num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(cfg.seed),
    ).frames[0]

    export_to_video(frames, str(out_path), fps=cfg.fps)
    return I2VResult(
        shot_id=scene.id or "shot", path=out_path, num_frames=cfg.num_frames,
        fps=cfg.fps, seconds=clip_seconds(cfg.num_frames, cfg.fps), distilled=cfg.distill,
    )


def _set_distill(pipe, distill: bool) -> None:
    """Toggle the Wan2.2-Lightning distill LoRA the ModelServer attached: active for the few-step
    path, scaled to zero for a full-step final render. Tolerant of a pipeline that never loaded
    the distill LoRA (the #12535 path), where full-step is simply the default."""
    try:
        if distill:
            pipe.set_adapters(["distill"], adapter_weights=[1.0])
        else:
            pipe.set_adapters(["distill"], adapter_weights=[0.0])
    except Exception:
        pass  # no distill adapter loaded -> already running full-step
