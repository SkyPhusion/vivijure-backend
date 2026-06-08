"""Per-scene SDXL keyframe: the still each shot's i2v animates from.

One scene, one image. For a single character the job is ordinary: SDXL with the slot's trained
LoRA and an IP-Adapter pulling identity from the bundle refs. The hard case is two characters
in one frame, where the naive result is *bleed*: one identity's features smear onto the other
(both faces drift toward an average, hair color crosses over). The fix this module implements
is regional: each character is bound to its own half of the canvas with a per-region IP-Adapter
mask and its own LoRA, at deliberately moderate scales (a strong per-slot scale is what causes
the cross-contamination), with optional OpenPose conditioning to plant two distinct bodies so
the regions have something to attach to.

Clean-room: the SDXL + IP-Adapter masking + ControlNet wiring is built from diffusers' own
documented interfaces (the IP-Adapter regional/masking guide, the SDXL ControlNet pipeline), not
any prior pipeline. The prompt building, region geometry, and engine-path decision are pure and
unit-tested on a CPU box; the generation body defers torch/diffusers/PIL and is validated on a
pod. Engine knobs live in `KeyframeParams` here; the control plane's typed `KeyframeConfig`
(separate work) maps into them, it does not replace them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .contract import Cast, Scene, Storyboard

# Anti-bleed defaults: a character LoRA pushed hard in a shared frame is exactly what bleeds, so
# the per-slot scales are deliberately moderate and the identity comes mostly from the masked
# IP-Adapter. Mirrors orchestrator.MULTI_CHAR_DEFAULTS (the planner's view of the same numbers).
DEFAULT_LORA_SCALE = 0.3
DEFAULT_IP_ADAPTER_SCALE = 0.7
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, extra limbs, fused faces, two heads, deformed, blurry, watermark, text"
)


@dataclass
class KeyframeParams:
    """Engine knobs for one keyframe render. Defaults suit the few-step distilled base (the
    keyframe gets animated, so 6-8 steps is plenty); the control plane's KeyframeConfig fills
    these per job."""
    steps: int = 8
    guidance_scale: float = 5.0
    resolution: int = 1024
    seed: int = 0
    few_step: bool = True            # use the Hyper-SD/DMD2 distilled path the ModelServer loaded
    lora_scale: float = DEFAULT_LORA_SCALE
    ip_adapter_scale: float = DEFAULT_IP_ADAPTER_SCALE
    pose_conditioning: bool = True   # ControlNet-OpenPose to separate bodies in multi-char shots
    negative_prompt: str = DEFAULT_NEGATIVE
    max_slots: int = 2               # regional path is tuned for 2; more is future work


@dataclass
class KeyframeResult:
    shot_id: str
    path: Path
    slots: list[str]
    multi_char: bool
    prompt: str
    seed: int
    engine: str                      # "single" | "regional"


# ------------------------------------------------------------------------- prompt building

def slot_trigger(cast: Cast, slot: str) -> str:
    """The token that summons a slot's identity in the prompt: its trained LoRA trigger, which
    is the character's name (see lora_train.TrainedLora.trigger), falling back to the slot id."""
    char = cast.characters.get(slot)
    return (char.name if char and char.name else slot)


def build_prompt(scene: Scene, cast: Cast, storyboard: Storyboard) -> str:
    """Compose the SDXL prompt: the storyboard style prefix, the scene's own prompt, and the
    trigger token for each character in the shot. Pieces are comma-joined and de-duplicated of
    empty fragments so a missing style or prompt never leaves a dangling comma."""
    triggers = ", ".join(slot_trigger(cast, s) for s in scene.character_slots)
    parts = [storyboard.style_prefix, scene.prompt, triggers]
    if storyboard.style_preset and storyboard.style_preset != "None":
        parts.append(storyboard.style_preset)
    # Fragments may carry their own edge commas (the control plane's style_prefix ends in ","),
    # so strip leading/trailing commas per fragment before joining; internal commas (the trigger
    # list) are untouched, so two characters never collapse into one.
    cleaned = [c for c in (p.strip().strip(",").strip() for p in parts if p) if c]
    return ", ".join(cleaned)


# ------------------------------------------------------------------------- region geometry

def region_boxes(width: int, height: int, n: int, *, orientation: str = "vertical") -> list[tuple[int, int, int, int]]:
    """Split the canvas into `n` equal regions, one per character, as (left, top, right, bottom)
    boxes. Vertical orientation (side-by-side strips) is the default for the common two-shot;
    horizontal stacks them. Pure geometry, so the mask generation that consumes it is testable
    without an image library."""
    if n <= 1:
        return [(0, 0, width, height)]
    boxes = []
    if orientation == "vertical":
        step = width // n
        for i in range(n):
            left = i * step
            right = width if i == n - 1 else (i + 1) * step  # last region absorbs the remainder
            boxes.append((left, 0, right, height))
    else:
        step = height // n
        for i in range(n):
            top = i * step
            bottom = height if i == n - 1 else (i + 1) * step
            boxes.append((0, top, width, bottom))
    return boxes


def engine_for(scene: Scene, params: KeyframeParams) -> str:
    """Which path this scene takes: 'regional' (the masked multi-identity, anti-bleed path) for a
    two-plus-character shot within the slot cap, else 'single'."""
    n = len(scene.character_slots)
    return "regional" if 2 <= n <= params.max_slots else "single"


# --------------------------------------------------------------------------- render (GPU)

def render_keyframe(
    scene: Scene,
    cast: Cast,
    storyboard: Storyboard,
    server,
    out_path: Path,
    *,
    params: KeyframeParams | None = None,
    lora_paths: dict[str, Path] | None = None,
    pose_image: "Path | None" = None,
) -> KeyframeResult:
    """Render one scene's keyframe to `out_path`.

    `server` is a `models.ModelServer` (provides the fp8 SDXL pipeline with the distill LoRA);
    `lora_paths` maps slot -> trained adapter file. Single-character shots take the ordinary
    identity path; two-character shots take the regional anti-bleed path. Heavy imports are
    deferred; the body is validated on a pod.
    """
    cfg = params or KeyframeParams()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(scene, cast, storyboard)
    slots = list(scene.character_slots)
    engine = engine_for(scene, cfg)

    import torch  # deferred: keep this module CPU-importable

    pipe = server.keyframe_pipeline()
    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
    loras = {s: Path(p) for s, p in (lora_paths or {}).items()}

    if engine == "single":
        image = _render_single(pipe, prompt, scene, cast, cfg, loras, generator)
    else:
        image = _render_regional(pipe, prompt, scene, cast, cfg, loras, generator, pose_image)

    image.save(out_path)
    return KeyframeResult(shot_id=scene.id or "shot", path=out_path, slots=slots,
                          multi_char=(engine == "regional"), prompt=prompt, seed=cfg.seed, engine=engine)


def _render_single(pipe, prompt, scene, cast, cfg, loras, generator):
    """One character (or none): the slot's LoRA bound, identity from a single IP-Adapter image.
    The plain SDXL identity path."""
    slot = scene.character_slots[0] if scene.character_slots else None
    _bind_loras(pipe, {slot: loras[slot]} if (slot and slot in loras) else {}, cfg.lora_scale)

    ip_images = _ref_images(cast, slot, count=1) if slot else None
    if ip_images:
        _ensure_ip_adapter(pipe)
        pipe.set_ip_adapter_scale(cfg.ip_adapter_scale)
    return pipe(
        prompt=prompt, negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        height=cfg.resolution, width=cfg.resolution, generator=generator,
        **({"ip_adapter_image": ip_images[0]} if ip_images else {}),
    ).images[0]


def _render_regional(pipe, prompt, scene, cast, cfg, loras, generator, pose_image):
    """Two characters, no bleed: each slot's identity is confined to its own region with a
    per-region IP-Adapter mask, each LoRA bound at the moderate per-slot scale, and (when on)
    OpenPose conditioning planting two bodies. The masks are what keep one face off the other."""
    from diffusers.image_processor import IPAdapterMaskProcessor

    slots = scene.character_slots[:cfg.max_slots]
    res = cfg.resolution
    boxes = region_boxes(res, res, len(slots), orientation="vertical")

    _bind_loras(pipe, {s: loras[s] for s in slots if s in loras}, cfg.lora_scale)

    # One IP-Adapter image per slot (its refs), each confined to its region's mask.
    _ensure_ip_adapter(pipe, n=len(slots))
    pipe.set_ip_adapter_scale([cfg.ip_adapter_scale] * len(slots))
    ip_images = [_ref_images(cast, s, count=1)[0] for s in slots]
    masks = IPAdapterMaskProcessor().preprocess(
        [_box_mask(res, res, b) for b in boxes], height=res, width=res)

    call_kwargs = dict(
        prompt=prompt, negative_prompt=cfg.negative_prompt,
        num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
        height=res, width=res, generator=generator,
        ip_adapter_image=ip_images,
        cross_attention_kwargs={"ip_adapter_masks": masks},
    )
    if cfg.pose_conditioning and pose_image is not None:
        from PIL import Image
        call_kwargs["image"] = Image.open(pose_image).convert("RGB")  # ControlNet-OpenPose hint
    return pipe(**call_kwargs).images[0]


# --------------------------------------------------------------------------- internals (GPU)

def _bind_loras(pipe, slot_paths: dict, scale: float) -> list[str]:
    """Load each slot's character LoRA and activate it alongside whatever base adapter is already
    on the pipe (a few-step distill LoRA, if the ModelServer loaded one). Character LoRAs go on at
    the moderate per-slot `scale` (the anti-bleed weight); any pre-existing base adapter stays at
    1.0. Discovering the base adapter instead of hardcoding a name means this is correct whether or
    not a distill LoRA is present, and a missing one never silently drops the character identity."""
    base = _adapter_names(pipe)  # adapters already on the pipe before we add characters
    loaded = []
    for slot, path in slot_paths.items():
        pipe.load_lora_weights(str(path), adapter_name=slot)
        loaded.append(slot)
    if loaded:
        names = base + loaded
        pipe.set_adapters(names, adapter_weights=[1.0] * len(base) + [scale] * len(loaded))
    return loaded


def _adapter_names(pipe) -> list[str]:
    """Names of the LoRA adapters currently loaded on the pipeline, de-duplicated across its
    components. Empty when none are loaded."""
    try:
        listed = pipe.get_list_adapters()  # {component: [adapter_name, ...]}
    except Exception:
        return []
    seen: list[str] = []
    for names in listed.values():
        for n in names:
            if n not in seen:
                seen.append(n)
    return seen


def _ensure_ip_adapter(pipe, n: int = 1):
    """Attach the SDXL IP-Adapter once (n encoders for n regional identities). The repo id mirrors
    models.DEFAULT_SPECS[IP_ADAPTER]."""
    if getattr(pipe, "_vj_ip_loaded", 0) >= n:
        return
    from .models import DEFAULT_SPECS, ModelRole
    repo = DEFAULT_SPECS[ModelRole.IP_ADAPTER].repo_id
    pipe.load_ip_adapter(repo, subfolder="sdxl_models",
                         weight_name=["ip-adapter_sdxl.safetensors"] * n if n > 1 else "ip-adapter_sdxl.safetensors")
    pipe._vj_ip_loaded = n


def _ref_images(cast, slot, count: int):
    """The first `count` reference images for a slot, as PIL images (the IP-Adapter identity)."""
    from PIL import Image
    char = cast.characters.get(slot)
    if not char or not char.ref_paths:
        return []
    return [Image.open(p).convert("RGB") for p in char.ref_paths[:count]]


def _box_mask(width, height, box):
    """A white-in-box, black-elsewhere mask image for one region (the IP-Adapter attends only
    where the mask is white)."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask
