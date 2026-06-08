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


def _server():
    global _SERVER
    if _SERVER is None:
        from .models import ModelServer  # deferred: torch only loads on a real worker
        _SERVER = ModelServer()
    return _SERVER


def build_pipeline(req: RenderRequest) -> GpuPipeline:
    """The GPU pipeline for one job: the job's typed config + its pretrained-LoRA references,
    over the shared model server."""
    return GpuPipeline(config=req.config, pretrained_loras=req.pretrained_loras, server=_server())


def handler(job: dict) -> dict:
    """RunPod serverless entry. Build the per-job pipeline from the request's RenderConfig,
    register it, and hand off to the harness (which owns model mirror, R2, plan, finish)."""
    from .harness.handler import handler as harness_handler

    payload = job.get("input", job)
    register_pipeline(build_pipeline(RenderRequest.from_dict(payload)))
    return harness_handler(job)
