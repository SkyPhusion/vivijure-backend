# Changelog

Notable changes per release. Releases are tagged `backend-vX.Y.Z` (SemVer-style,
pre-1.0: PATCH for fixes and backend-only tweaks, MINOR for new features). Entries are
newest-first. History before this file was introduced lives in the git tags; the recent
releases are summarized below from that history.

## Unreleased

**Wire the few-step keyframe distill into the GPU path (the speed lever).** The tier config
already asked draft/standard for a 4/8-step, cfg=0, DDIM-trailing keyframe, but the pipeline
never loaded the Hyper-SD distill LoRA, never set the scheduler, and the `few_step` flag was
dead, so draft previews ran few-step *without* the LoRA that makes few-step work (degraded
stills). Now the ModelServer loads the Hyper-SDXL distill LoRA as a persistent base adapter
("distill"), `keyframe._bind_loras` gates its weight on the tier (1.0 on draft/standard, 0.0
on final, so one warm pipe serves every tier with no reload), and `keyframe._apply_scheduler`
pins DDIM-trailing for the few-step path and restores a full-step solver for final. The final
tier is untouched; this speeds up previews (the most-repeated op). GPU-validation pending: the
single 8-step distill LoRA driving a 4-step draft is the thing to eyeball.

Code: `models.py` (ModelSpec.weight_name; load the distill LoRA in `keyframe_pipeline`),
`keyframe.py` (DISTILL_ADAPTER, `KeyframeParams.scheduler`, `_apply_scheduler`, few-step gating
in `_bind_loras`), `pipeline.py` (thread `scheduler` through `keyframe_params_from`),
`tests/test_keyframe.py` + `tests/test_pipeline.py` (distill-weight + scheduler mapping).
Full suite green.

Public-readiness housekeeping ahead of open-sourcing the repository: added
`CONTRIBUTING.md` (clean-room / independence-protective posture, house rules, DCO),
`SECURITY.md` (private vulnerability reporting + the render-backend security boundary),
`CODE_OF_CONDUCT.md`, and this `CHANGELOG.md`. Corrected the README architecture table to
reflect the shipped pipeline, and removed em-dashes from the README to match the house
style.

## backend-v0.1.15

Multi-character pose: wire the OpenPose ControlNet so a 2+ character shot plants two
distinct bodies instead of a blended one (`keyframe.py`).

## backend-v0.1.14

Stamp `user_email` on uploaded artifacts so the control plane's `/api/artifact` ownership
check can serve them back (`harness`).

## backend-v0.1.13

Fail loud when a staged LoRA registers no adapter, and make reused-LoRA injection
fail-fast too, so a silent no-op never ships a render without the intended character
(`keyframe.py`, `pipeline.py`).

## backend-v0.1.12

First-class `preview` action for keyframes-only renders: the orchestrator short-circuits
after the SDXL pass when only keyframes are requested (`orchestrator.py`).

## backend-v0.1.11

Cache both Wan 2.2 MoE experts to clear the step-12 hit-cliff in image-to-video
(`i2v.py`).

## backend-v0.1.10

Use the matched `enable_cache` / `disable_cache` pair for the i2v feature cache (`i2v.py`).

## backend-v0.1.2 -- backend-v0.1.9

Earlier pipeline build-out: the structured render progress channel (R2-backed), the
feature-cache denoise accelerator, reused-LoRA staging from R2 so warm workers skip
retraining, the R2-mirror model loader, and the AGPL-3.0-only license. See the git tags
for the per-release detail.
