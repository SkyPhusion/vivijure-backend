"""vivijure-render: a clean-room render backend for Vivijure."""
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
]
