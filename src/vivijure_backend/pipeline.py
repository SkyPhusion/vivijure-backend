"""GpuPipeline: the convergence. Wire the five engines into one render.

The harness decides the I/O and the order (bundle in -> plan -> `pipeline.execute` -> finish ->
out); the orchestrator decides, on the CPU, exactly what work survives (which LoRAs to train,
which keyframes to draw, which shots to animate). This module is the glue between that plan and
the GPU engines: it trains the kept LoRAs, draws the GENERATE keyframes with them, animates the
needs_i2v shots, and reuses everything the plan said to reuse, all on ONE shared `ModelServer`
so models load once per worker.

The typed `RenderConfig` drives the engines through two pure mappers (`keyframe_params_from`,
`i2v_params_from`): the control plane's config in, the engines' `KeyframeParams` / `I2VParams`
out. The three GPU stages sit behind small overridable methods so the orchestration is testable
on a CPU box with the engines stubbed (the same fake-stage pattern `tests/test_harness.py` uses
for the `Pipeline` protocol).

Clean-room: built only from our own modules (config / orchestrator / keyframe / i2v / lora_train
/ harness) and their documented signatures; no fork.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import i2v as _i2v
from . import keyframe as _keyframe
from . import lora_train as _lora_train
from .config import RenderConfig
from .harness.handler import Outputs
from .harness.progress import NullEmitter
from .keyframe import KeyframeParams, build_prompt
from .i2v import I2VParams, frames_for
from .orchestrator import KeyframeMode, RenderPlan
from .contract import Bundle

# Shared no-op emitter for a pipeline run with no progress channel wired (the test default).
_NULL_PROGRESS = NullEmitter()


# --------------------------------------------------------------- config -> engine params

def keyframe_params_from(config: RenderConfig) -> KeyframeParams:
    """Map the typed `KeyframeConfig` onto the keyframe engine's `KeyframeParams`. `distill`
    selects the few-step path (its `distill_steps`) and `few_step`; the per-slot anti-bleed
    scales come from the multi_char block (the engine uses one IP-Adapter / LoRA scale for both
    the single and the masked-regional path, which is what KeyframeParams models)."""
    kc = config.keyframe
    mc = kc.multi_char
    return KeyframeParams(
        steps=kc.distill_steps if kc.distill else kc.steps,
        guidance_scale=kc.guidance_scale,
        resolution=kc.width,                       # engine renders square; width is the side
        seed=kc.seed,
        few_step=kc.distill,
        scheduler=kc.scheduler.value,              # ddim_trailing on the few-step path, a solver on final
        lora_scale=mc.lora_scale_per_slot,
        ip_adapter_scale=mc.ip_adapter_scale_per_slot,
        pose_conditioning=mc.pose_conditioning,
        controlnet_pose_scale=mc.controlnet_pose_scale,
        region_gutter=mc.region_gutter,
        max_slots=mc.max_slots,
    )


def i2v_params_from(config: RenderConfig, scene) -> I2VParams:
    """Map the typed `I2VConfig` (+ the scene's duration) onto the i2v engine's `I2VParams`.
    Frame count is derived from the scene's target seconds at the config fps (snapped to the
    temporal VAE's 4k+1 by `i2v.frames_for`); `distill` selects the 4-step Lightning path and its
    step/guidance profile; `feature_cache` carries the tier's denoise accelerator (final=MIXCACHE,
    standard=EASYCACHE, draft=NONE; `from_dict` already force-clears it to NONE under distill, so the
    engine never sees a cache on a 4-step render). `flow_shift` / `loader` are still not surfaced by
    `I2VParams` (lower-priority gaps), so they remain unmapped."""
    ic = config.i2v
    p = I2VParams(
        num_frames=frames_for(scene.target_seconds, ic.fps),
        fps=ic.fps,
        steps=ic.distill_steps if ic.distill else ic.steps,
        guidance_scale=ic.guidance_scale,
        distill=ic.distill,
        seed=config.keyframe.seed,                 # one seed for the render; i2v has no own seed knob
        feature_cache=ic.feature_cache,
    )
    if ic.negative_prompt:
        p.negative_prompt = ic.negative_prompt
    return p


# --------------------------------------------------------------------------- the pipeline

@dataclass
class GpuPipeline:
    """The deployed GPU `Pipeline`. Built from a job's `RenderConfig`; holds the shared
    `ModelServer` so a warm worker loads each model once. `pretrained_loras` (slot -> reference)
    lets a prior adapter feed keyframing when it is staged locally."""
    config: RenderConfig
    pretrained_loras: dict[str, str] = field(default_factory=dict)
    server: Any = None  # models.ModelServer; created lazily on first GPU use

    def _model_server(self):
        if self.server is None:
            from .models import ModelServer  # deferred: keep this module CPU-importable
            self.server = ModelServer()
        return self.server

    def set_progress(self, emitter) -> None:
        """Wire a progress emitter (the harness calls this per job). Default is a no-op emitter, so
        the pipeline runs unchanged without one."""
        self._progress = emitter

    @property
    def progress(self):
        return getattr(self, "_progress", None) or _NULL_PROGRESS

    def set_pretrained_loras(self, mapping: dict[str, str]) -> None:
        """Replace the reused-LoRA refs with the harness's local-path map (it stages them from R2
        before execute). The pipeline never touches R2 itself; it just loads the local files the
        `if p.is_file()` check in `execute` already understands."""
        self.pretrained_loras = dict(mapping)

    # --- GPU stages, behind overridable methods (stubbed in CPU tests) ---

    def _train_slot(self, char, out_dir: Path) -> Path:
        # Throttled per-step training progress (the long pole); lora_train calls the cb every N steps.
        cb = self.progress.train_step_cb(char.slot)
        return _lora_train.train_slot(char, out_dir, config=self.config.lora, progress_cb=cb).path

    def _render_keyframe(self, scene, cast, storyboard, out_path: Path, lora_paths: dict[str, Path]) -> Path:
        return _keyframe.render_keyframe(
            scene, cast, storyboard, self._model_server(), out_path,
            params=keyframe_params_from(self.config), lora_paths=lora_paths,
        ).path

    def _animate(self, scene, keyframe_path: Path, prompt: str, out_path: Path) -> Path:
        # Per-step i2v progress (every step; i2v is 4-40 steps, ~30s/step at final tier).
        cb = self.progress.i2v_step_cb(scene.id)
        return _i2v.animate(
            scene, keyframe_path, prompt, self._model_server(), out_path,
            params=i2v_params_from(self.config, scene), progress_cb=cb,
        ).path

    # --- orchestration (CPU; the stages above are the only GPU touch points) ---

    def execute(self, plan: RenderPlan, bundle: Bundle, workdir: Path) -> Outputs:
        out = Outputs()
        workdir = Path(workdir)
        cast, storyboard = bundle.cast, bundle.storyboard
        scenes_by_id = {s.id: s for s in storyboard.scenes}

        # 1) Train the LoRAs the plan kept; collect adapter paths for keyframing.
        lora_paths: dict[str, Path] = {}
        for slot in plan.lora.train:
            char = cast.characters.get(slot)
            if char is None:
                continue  # plan listed a slot the cast does not define; nothing to train
            path = self._train_slot(char, workdir / "loras" / slot)
            out.loras[slot] = path
            lora_paths[slot] = path
            self.progress.emit("train_done", slot=slot, path=str(path))
        # Reused / pretrained adapters feed keyframing too, when staged on disk locally (the
        # adapter is portable .safetensors; an R2-key reference that is not a local file is left
        # to the deploy to stage, and the shot falls back to IP-Adapter identity if absent).
        for slot, ref in self.pretrained_loras.items():
            p = Path(ref)
            if p.is_file():
                lora_paths.setdefault(slot, p)

        # 2) Per scene: draw the keyframe (or resolve a reused/injected one), then animate.
        for sp in plan.scenes:
            scene = scenes_by_id.get(sp.shot_id)
            if scene is None:
                continue
            if sp.keyframe_mode is KeyframeMode.GENERATE:
                kf_path = self._render_keyframe(
                    scene, cast, storyboard, workdir / "keyframes" / f"{sp.shot_id}.png", lora_paths)
                out.keyframes[sp.shot_id] = kf_path
                self.progress.emit("keyframe_done", shot=sp.shot_id)
            else:
                kf_path = self._resolve_keyframe(sp, scene, bundle, workdir)
            if sp.needs_i2v and kf_path is not None:
                clip = self._animate(
                    scene, kf_path, build_prompt(scene, cast, storyboard),
                    workdir / "clips" / f"{sp.shot_id}.mp4")
                out.clips.append((sp.shot_id, clip))
                self.progress.emit("i2v_done", shot=sp.shot_id)
        return out

    def _resolve_keyframe(self, sp, scene, bundle: Bundle, workdir: Path) -> Path | None:
        """The keyframe to animate when the plan did not (re)generate it: the authored
        `start_image` for an INJECT shot, or a keyframe a prior pass already left on disk for a
        REUSE shot. Returns None if nothing is staged, so the shot is skipped rather than crashing
        the whole render."""
        if sp.keyframe_mode is KeyframeMode.INJECT and scene.start_image:
            cand = bundle.root / scene.start_image
            return cand if cand.is_file() else None
        for cand in (
            workdir / "keyframes" / f"{sp.shot_id}.png",
            bundle.root / "keyframes" / f"{sp.shot_id}.png",
        ):
            if cand.is_file():
                return cand
        return None
