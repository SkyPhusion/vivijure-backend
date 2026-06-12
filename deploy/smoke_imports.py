"""Import smoke for finishing-stage runtime deps.

Run inside the built Docker image (CPU-only; no GPU / model weights needed) to verify every
required package is present BEFORE the image is pushed to GHCR:

    docker run --rm <image> conda run --no-capture-output -n vivijure python /opt/vivijure/smoke_imports.py

Exit 0 = all imports OK. Non-zero = at least one dep missing; FAIL lines on stderr identify what.
A dep missing here fails CI in seconds rather than crashing the worker after a 33-min GPU render.

Closes the loop with issue #9 (CI coverage of the runtime dep surface).
"""
import importlib
import sys

CHECKS = [
    ("av",                                       "PyAV: imageio pyav plugin for finish_clip"),
    ("gfpgan",                                   "GFPGAN blind face restorer"),
    ("basicsr.utils.registry",                   "basicsr ARCH_REGISTRY (codeformer path)"),
    ("facexlib.utils.face_restoration_helper",   "facexlib face detection helper"),
    ("rife.RIFE_HDv3",                           "vendored RIFE HDv3 frame interpolator (Model loader)"),
    ("vivijure_backend.finish",                  "finishing stage (must stay CPU-importable)"),
]

failed = []
for mod, label in CHECKS:
    try:
        importlib.import_module(mod)
        print(f"OK    {mod}  ({label})")
    except Exception as exc:
        print(f"FAIL  {mod}  ({label}): {exc}", file=sys.stderr)
        failed.append(mod)

if failed:
    print(f"\n{len(failed)} import(s) failed; see FAIL lines above.", file=sys.stderr)
    sys.exit(len(failed))

print(f"\nAll {len(CHECKS)} finish-stage imports OK.")
