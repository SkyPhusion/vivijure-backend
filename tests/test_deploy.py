"""CPU tests for the deploy plumbing: the pure RunPod template-pin transform, and the worker
entry's per-job pipeline build over a shared model server. No Docker, no network, no GPU."""
import importlib.util
from pathlib import Path

from vivijure_backend import worker
from vivijure_backend.contract import RenderRequest
from vivijure_backend.pipeline import GpuPipeline


def _load_pin_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pin-runpod-template.py"
    spec = importlib.util.spec_from_file_location("pin_runpod_template", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------- pin-runpod-template helpers

def test_prepare_template_splices_image_and_strips_typename():
    pin = _load_pin_script()
    fetched = {
        "__typename": "PodTemplate", "id": "tpl1", "name": "vj-backend",
        "imageName": "ghcr.io/skyphusion-labs/vivijure-backend:0.1.0",
        "env": [{"__typename": "EnvVar", "key": "HF_HOME", "value": "/opt/models/hf-cache"}],
    }
    out = pin.prepare_template(fetched, "ghcr.io/skyphusion-labs/vivijure-backend:0.2.0")
    assert out["imageName"] == "ghcr.io/skyphusion-labs/vivijure-backend:0.2.0"
    assert "__typename" not in out
    assert "__typename" not in out["env"][0]          # stripped recursively
    assert out["env"][0]["key"] == "HF_HOME"           # real fields preserved
    # the caller's dict is not mutated (image still the old tag, typename intact)
    assert fetched["imageName"].endswith(":0.1.0")
    assert fetched["__typename"] == "PodTemplate"


def test_strip_typename_handles_nested_lists_and_dicts():
    pin = _load_pin_script()
    o = {"__typename": "A", "xs": [{"__typename": "B", "k": 1}, {"k": 2}]}
    pin.strip_typename(o)
    assert "__typename" not in o and "__typename" not in o["xs"][0]


# ----------------------------------------------------------------------------- worker entry

def _req(**over):
    return RenderRequest.from_dict({"action": "render", "project": "neon", "bundle_key": "x",
                                    "quality_tier": "draft", **over})


def test_build_pipeline_carries_job_config_and_pretrained():
    req = _req(pretrained_loras={"A": "loras/ext/A.safetensors"})
    pipe = worker.build_pipeline(req)
    assert isinstance(pipe, GpuPipeline)
    assert pipe.config is req.config
    assert pipe.config.quality.value == "draft"
    assert pipe.pretrained_loras == {"A": "loras/ext/A.safetensors"}


def test_build_pipeline_shares_one_model_server_across_jobs():
    # Warm-worker reuse: every per-job pipeline wraps the SAME process-global ModelServer.
    a = worker.build_pipeline(_req())
    b = worker.build_pipeline(_req(quality_tier="final"))
    assert a.server is b.server is not None
    assert a.config is not b.config          # but each carries its own job config
