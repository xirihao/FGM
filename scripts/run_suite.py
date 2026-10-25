"""Replay benchmark caches with fresh adaptation state for every run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from fgm.io import write_json

TEN = ["oxford_flowers", "dtd", "oxford_pets", "stanford_cars", "ucf101", "caltech101", "food101", "sun397", "fgvc", "eurosat"]
FIVE = ["imagenet", "imagenet-a", "imagenet-v", "imagenet-r", "imagenet-s"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, required=True, help="One subdirectory per dataset")
    p.add_argument("--suite", choices=["ten", "imagenet", "all"], required=True)
    p.add_argument("--labels-root", type=Path, help="Optional directory containing <dataset>/labels.npy")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--order-seeds", nargs="+", type=int, help="Omit to preserve the cached order")
    p.add_argument("--config", type=Path, default=Path("configs/default.json"))
    p.add_argument("--normalization", choices=["guarded", "legacy"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--uniform-h", type=float, help="Use this h on every dataset, overriding the Caltech101 exception")
    args = p.parse_args()
    names = TEN if args.suite == "ten" else FIVE if args.suite == "imagenet" else TEN + FIVE
    orders = args.order_seeds or [None]
    if args.order_seeds and len(set(orders)) != len(orders):
        p.error("Order seeds must be distinct")
    scripts = Path(__file__).resolve().parent
    results = {}
    for name in names:
        results[name] = []
        for seed in orders:
            run_name = "stored" if seed is None else f"order_{seed}"
            out = args.output / name / f"{run_name}.npz"
            cmd = [sys.executable, str(scripts / "replay.py"), "--cache", str(args.cache_root / name),
                   "--output", str(out), "--config", str(args.config), "--device", args.device, "--dtype", args.dtype]
            if args.uniform_h is not None:
                cmd += ["--h", str(args.uniform_h)]
            elif name == "caltech101":
                cmd += ["--h", "50"]
            if seed is not None:
                cmd += ["--order-seed", str(seed)]
            if args.normalization:
                cmd += ["--normalization", args.normalization]
            subprocess.run(cmd, check=True)
            if args.labels_root:
                score = out.with_name(out.stem + "_accuracy.json")
                subprocess.run([sys.executable, str(scripts / "score.py"), "--predictions", str(out),
                                "--labels", str(args.labels_root / name / "labels.npy"), "--output", str(score)], check=True)
                results[name].append(json.loads(score.read_text())["accuracy"])
    if args.labels_root:
        summary = {"datasets": {}, "aggregate": {}, "order_seeds": orders}
        branches = next(iter(results.values()))[0].keys()
        for name, runs in results.items():
            summary["datasets"][name] = {branch: {"runs": [r[branch] for r in runs],
                "mean": float(np.mean([r[branch] for r in runs])),
                "sample_std": float(np.std([r[branch] for r in runs], ddof=1)) if len(runs) > 1 else None} for branch in branches}
        groups = {"dataset_mean": names}
        if all(name in names for name in FIVE):
            groups["imagenet_mean"] = FIVE
            groups["ood_mean"] = FIVE[1:]
        if all(name in names for name in TEN):
            groups["ten_dataset_mean"] = TEN
        for group, members in groups.items():
            summary["aggregate"][group] = {}
            for branch in branches:
                means = [float(np.mean([results[name][j][branch] for name in members])) for j in range(len(orders))]
                summary["aggregate"][group][branch] = {"runs": means, "mean": float(np.mean(means)),
                    "sample_std": float(np.std(means, ddof=1)) if len(means) > 1 else None}
        write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
