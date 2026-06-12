"""RunPod worker entry point: build the GPU pipeline for a job and run it.

This is the module the worker image runs. For each job it builds a `GpuPipeline` from the job's
typed `RenderConfig`, registers it on the harness seam, and delegates to the harness's `handler`
(which mirrors models on a cold worker, opens R2, plans on the CPU, and drives the pipeline). The
`ModelServer` is a process global so a warm worker reuses loaded models across jobs; only the
lightweight per-job pipeline wrapper is rebuilt, carrying that job's config.

CPU-importable: the torch-touching `ModelServer` is created lazily on first real GPU use, so this
module (and `build_pipeline`) import and test without a GPU.
"""
from __future__ import annotations

from .contract import RenderRequest
from .harness.pipeline_registry import register_pipeline
from .pipeline import GpuPipeline

_SERVER = None  # process-global ModelServer; warm workers reuse loaded models across jobs


def _server(req: RenderRequest | None = None):
    """Return the process-global ModelServer, building it on first call.

    Model loading is expensive and models are shared across jobs on a warm worker, so the server
    is a singleton. On first call, specs from `req.config` (keyframe base/distill, i2v base/distill)
    are wired in so a non-default model repo configured in the job DOES take effect -- provided this
    is the cold-start job. On subsequent calls (warm worker, server already built) the existing
    model set is reused regardless of `req.config.*.model` fields; per-job hot-swapping requires a
    pod restart. If `req` is None the server uses DEFAULT_SPECS (e.g. standalone test usage)."""
    global _SERVER
    if _SERVER is None:
        from .models import ModelServer, ModelSpec, ModelRole, DEFAULT_SPECS  # deferred: torch
        job_specs: dict = {}
        if req is not None:
            kc, ic = req.config.keyframe, req.config.i2v
            D = DEFAULT_SPECS
            job_specs = {
                ModelRole.KEYFRAME_BASE: ModelSpec(
                    ModelRole.KEYFRAME_BASE, kc.base_model, D[ModelRole.KEYFRAME_BASE].family),
                ModelRole.KEYFRAME_FEWSTEP: ModelSpec(
                    ModelRole.KEYFRAME_FEWSTEP, kc.distill_model, D[ModelRole.KEYFRAME_FEWSTEP].family),
                ModelRole.I2V: ModelSpec(
                    ModelRole.I2V, ic.model, D[ModelRole.I2V].family),
                ModelRole.I2V_DISTILL: ModelSpec(
                    ModelRole.I2V_DISTILL, ic.distill_model, D[ModelRole.I2V_DISTILL].family),
            }
        _SERVER = ModelServer(specs=job_specs or None)
    return _SERVER


def build_pipeline(req: RenderRequest) -> GpuPipeline:
    """The GPU pipeline for one job: the job's typed config + its pretrained-LoRA references,
    over the shared model server. The server is initialized from `req.config` model fields on
    the first (cold-start) job; subsequent jobs reuse the already-loaded model set."""
    return GpuPipeline(config=req.config, pretrained_loras=req.pretrained_loras, server=_server(req))


def handler(job: dict) -> dict:
    """RunPod serverless entry. Build the per-job pipeline from the request's RenderConfig,
    register it, and hand off to the harness (which owns model mirror, R2, plan, finish)."""
    from .harness.handler import handler as harness_handler

    payload = job.get("input", job)
    register_pipeline(build_pipeline(RenderRequest.from_dict(payload)))
    return harness_handler(job)


def main() -> None:
    """The container's main process: start the RunPod serverless loop with our handler. The
    `runpod` SDK import is deferred so this module stays CPU/dep-light for tests; the worker
    image installs it. `python -m vivijure_backend.worker` runs this."""
    import runpod  # the worker image's serverless SDK

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
