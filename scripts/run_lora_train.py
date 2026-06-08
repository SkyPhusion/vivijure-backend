#!/usr/bin/env python3
"""Pod entry point for a LoRA training run: extract a bundle, train one (or every) character
slot from its refs, write the adapters, print a JSON summary.

    python scripts/run_lora_train.py BUNDLE.tar.gz OUT_DIR [--slot A] [--steps 1000]

Run on a CUDA pod with torch/diffusers/transformers/peft installed. The control-plane harness
will call `train_slot` directly; this is the standalone build-and-run driver.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vivijure_backend.contract import Bundle  # noqa: E402
from vivijure_backend.lora_train import LoraTrainConfig, train_slot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--slot", help="train only this slot; default trains every use_characters slot")
    ap.add_argument("--steps", type=int, default=LoraTrainConfig.max_steps)
    ap.add_argument("--rank", type=int, default=LoraTrainConfig.rank)
    ap.add_argument("--resolution", type=int, default=LoraTrainConfig.resolution)
    args = ap.parse_args()

    bundle = Bundle.extract(args.bundle, args.out_dir / "bundle")
    cfg = LoraTrainConfig(max_steps=args.steps, rank=args.rank, resolution=args.resolution)

    slots = [args.slot] if args.slot else list(bundle.storyboard.use_characters)
    results = []
    for slot in slots:
        char = bundle.cast.characters.get(slot)
        if char is None:
            print(f"skip {slot}: not in cast registry", file=sys.stderr)
            continue
        print(f"== training slot {slot} ({char.name}): {len(char.ref_paths)} refs, "
              f"{cfg.max_steps} steps, rank {cfg.rank} ==", flush=True)
        tl = train_slot(char, args.out_dir / "loras" / slot, config=cfg)
        results.append({
            "slot": tl.slot, "trigger": tl.trigger, "path": str(tl.path),
            "steps": tl.steps, "rank": tl.rank, "refs": tl.ref_count,
            "base": tl.base_repo, "meta": tl.meta,
            "exists": tl.path.is_file(), "bytes": tl.path.stat().st_size if tl.path.is_file() else 0,
        })

    print("\n=== RESULT ===")
    print(json.dumps(results, indent=2))
    return 0 if results and all(r["exists"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
