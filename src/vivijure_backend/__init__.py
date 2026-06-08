"""vivijure-backend: a clean-room render backend for Vivijure."""
from .contract import (
    Bundle,
    Cast,
    Character,
    Clip,
    Keyframe,
    RenderRequest,
    RenderResult,
    Scene,
    Storyboard,
)
from .device import Arch, Attention, Device, Quant, Tier, current
from .models import DEFAULT_SPECS, ModelFamily, ModelRole, ModelServer, ModelSpec, quant_for
from .orchestrator import Action, KeyframeMode, LoraPlan, RenderPlan, ScenePlan, plan, validate
from .routing import RUNPOD_GPU_ID, QualityTier, Stage, gpu_for

__all__ = [
    "Bundle",
    "Cast",
    "Character",
    "Clip",
    "Keyframe",
    "RenderRequest",
    "RenderResult",
    "Scene",
    "Storyboard",
    # device + routing
    "Arch",
    "Attention",
    "Device",
    "Quant",
    "Tier",
    "current",
    "Stage",
    "QualityTier",
    "gpu_for",
    "RUNPOD_GPU_ID",
    # models
    "ModelServer",
    "ModelRole",
    "ModelSpec",
    "ModelFamily",
    "quant_for",
    "DEFAULT_SPECS",
    # orchestrator (CPU planning)
    "plan",
    "validate",
    "RenderPlan",
    "ScenePlan",
    "LoraPlan",
    "Action",
    "KeyframeMode",
]
