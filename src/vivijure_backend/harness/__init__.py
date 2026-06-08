"""Serverless harness: the RunPod entry point, R2 I/O, and cold-start model mirror that
wrap the CPU planner and the GPU stages into a deployable worker.

The harness is deliberately thin. It downloads a bundle, asks the planner what to do, drives
the stages the plan did not eliminate, and pushes the results back to R2. The stages
themselves (LoRA train, keyframe, i2v, assemble) live behind a `Pipeline` protocol so this
layer carries no model code and imports on a CPU box; a concrete GPU pipeline is injected at
deploy time.
"""
