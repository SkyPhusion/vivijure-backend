"""The worker's job flow: bundle in -> plan -> stages -> finish -> results out.

This is the spine that turns a RunPod job into a render and back into R2 keys. It owns the I/O
contract (what comes in, what goes out, where) and the order of operations; it owns no model
code. The GPU stages sit behind the `Pipeline` protocol and are injected, so this module
imports and tests on a CPU box: `run_job` is exercised with a fake pipeline and a fake store,
and the real `handler` entry point wires the live R2 client, the cold-start model mirror, and
the deployed GPU pipeline.

The finish is deliberately off-GPU (see the planner's `assemble_off_gpu`): a normal render
merges the clips here with ffmpeg, while an offloaded finish (`finish_offloaded`) just uploads
the per-shot clips plus a manifest for a separate CPU container to merge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..assemble import ClipInput, assemble, build_manifest, order_for_storyboard, write_manifest
from ..contract import Bundle, RenderRequest, RenderResult, Keyframe, Clip
from ..orchestrator import RenderPlan, plan as make_plan, validate
from . import keys


class HarnessError(RuntimeError):
    """A job that failed in the harness layer (bad bundle, validation, missing stage output)."""


@dataclass
class Outputs:
    """What a `Pipeline.execute` produced on disk. The harness turns these into R2 objects.

    `clips` are (shot_id, path) the harness orders by the storyboard before merging. A pipeline
    that already merged the film can set `final_video`; otherwise the harness assembles it."""
    loras: dict[str, Path] = field(default_factory=dict)        # slot -> adapter file
    keyframes: dict[str, Path] = field(default_factory=dict)    # shot_id -> png
    clips: list[tuple[str, Path]] = field(default_factory=list)  # (shot_id, mp4)
    final_video: Path | None = None
    audio: Path | None = None


@runtime_checkable
class Pipeline(Protocol):
    """The GPU stages, injected. Given the plan and the extracted bundle, run only the work the
    plan did not eliminate (train the listed LoRAs, generate the GENERATE keyframes, animate the
    needs_i2v shots) and return where the artifacts landed. Implemented by the model layer; the
    harness never imports torch."""

    def execute(self, plan: RenderPlan, bundle: Bundle, workdir: Path) -> Outputs: ...


def run_job(
    job: dict,
    *,
    pipeline: Pipeline,
    store,
    workdir: Path,
    trained_slots: set[str] = frozenset(),
    existing_keyframes: set[str] = frozenset(),
) -> dict:
    """Run one render job end to end and return the control-plane response dict.

    `store` is an R2-like object with `get_file`, `put_file`, `put_dir_as_tar` (the real `R2`,
    or a fake in tests). Nothing here touches a GPU; the GPU work is `pipeline.execute`.
    """
    req = RenderRequest.from_dict(job)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    # --- bundle in ---
    tar = store.get_file(req.bundle_key, workdir / "bundle.tar.gz")
    bundle = Bundle.extract(Path(tar), workdir / "project")

    # --- validate + plan (CPU) ---
    errs = validate(req, bundle.storyboard)
    if errs:
        raise HarnessError("invalid render job: " + "; ".join(errs))
    plan = make_plan(
        req, bundle.storyboard,
        trained_slots=set(trained_slots) | set(req.pretrained_loras),
        existing_keyframes=set(existing_keyframes),
    )

    # --- GPU stages (only what the plan kept) ---
    outputs = pipeline.execute(plan, bundle, workdir)

    # --- finish + results out ---
    return _finish(req, plan, bundle, outputs, store, workdir).to_dict()


def _finish(req: RenderRequest, plan: RenderPlan, bundle: Bundle, outputs: Outputs,
            store, workdir: Path) -> RenderResult:
    project = req.project
    result = RenderResult(project=project)

    # LoRA adapters: upload trained ones, pass pretrained through.
    for slot, path in outputs.loras.items():
        key = store.put_file(Path(path), keys.lora_key(project, slot))
        result.lora[slot] = {"lora_id": key}
    for slot, lora_id in req.pretrained_loras.items():
        result.lora.setdefault(slot, {"lora_id": lora_id})

    # Keyframes: upload whatever the stage drew.
    for shot_id, path in outputs.keyframes.items():
        key = store.put_file(Path(path), keys.keyframe_key(project, shot_id), content_type="image/png")
        result.keyframes.append(Keyframe(shot_id=shot_id, key=key))

    # Clips ordered by the storyboard (never the stage's incidental order).
    ordered = order_for_storyboard(
        [ClipInput(shot_id=s, path=Path(p)) for s, p in outputs.clips], bundle.storyboard)

    offloaded = bool(req.overrides.get("finish_offloaded"))
    if offloaded:
        # Off-GPU finish elsewhere: emit per-shot clips + a manifest, no merge here.
        for c in ordered:
            key = store.put_file(c.path, keys.clip_key(project, c.shot_id), content_type="video/mp4")
            result.clips.append(Clip(shot_id=c.shot_id, key=key))
        manifest = build_manifest(ordered, output_name="full.mp4",
                                  audio=str(outputs.audio) if outputs.audio else None)
        man_path = write_manifest(manifest, workdir / "manifest.json")
        store.put_file(man_path, keys.join("renders", project, "manifest.json"),
                       content_type="application/json")
    elif ordered or outputs.final_video:
        # Normal finish: merge here (off-GPU) unless the pipeline already produced the film.
        final = Path(outputs.final_video) if outputs.final_video else \
            assemble(ordered, workdir / "full.mp4", audio=outputs.audio).output_path
        from ..assemble import probe_duration, probe_has_audio
        result.output_key = store.put_file(final, keys.output_key(project), content_type="video/mp4")
        result.seconds = probe_duration(final)
        result.has_audio = probe_has_audio(final)

    # Project state for the next incremental render.
    result.state_key = store.put_dir_as_tar(bundle.root, keys.state_key(project))
    return result


def handler(job: dict) -> dict:
    """RunPod serverless entry point. Mirrors models on a cold worker, builds the live R2
    client, runs the job through the deployed GPU pipeline, returns the response. RunPod passes
    `{"input": {...}}`; the render request is the inner dict."""
    import tempfile

    from .models_mirror import ensure_models
    from .r2 import R2, R2Config
    from .pipeline_registry import get_pipeline  # the deploy registers its GPU pipeline here

    ensure_models()
    store = R2(R2Config.from_env())
    payload = job.get("input", job)
    workdir = Path(tempfile.mkdtemp(prefix="vj-job-"))
    return run_job(payload, pipeline=get_pipeline(), store=store, workdir=workdir)
