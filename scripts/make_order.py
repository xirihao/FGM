"""Prepare a controlled stream order before running adaptation.

Class-concentrated protocols use labels here solely to construct the test stream.
The resulting permutation contains indices only. replay.py never reads labels.
"""
import argparse
from pathlib import Path
import numpy as np
from fggm.io import write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--mode", choices=["random", "dirichlet", "separate"], required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--concentration", type=float, default=0.1)
    p.add_argument("--blocks", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.suffix != ".npy" or args.output.exists():
        raise ValueError("Choose a new .npy output file")
    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 1 or not len(labels) or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty integer vector")
    if args.blocks < 1 or args.concentration <= 0:
        raise ValueError("blocks and concentration must be positive")
    rng = np.random.default_rng(args.seed)
    if args.mode == "random":
        order = rng.permutation(len(labels))
    elif args.mode == "separate":
        order = np.concatenate([rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)])
    else:
        blocks = [[] for _ in range(args.blocks)]
        for cls in np.unique(labels):
            rows = rng.permutation(np.flatnonzero(labels == cls))
            probabilities = rng.dirichlet(np.full(args.blocks, args.concentration))
            counts = rng.multinomial(len(rows), probabilities)
            start = 0
            for bucket, count in zip(blocks, counts):
                bucket.extend(rows[start:start + count].tolist())
                start += count
        order = np.asarray([row for bucket in blocks for row in rng.permutation(bucket)], dtype=np.int64)
    if not np.array_equal(np.sort(order), np.arange(len(labels))):
        raise RuntimeError("Generated order is not a complete permutation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, order)
    write_json(args.output.with_suffix(".json"), {"mode": args.mode, "seed": args.seed, "n": len(labels),
               "blocks": args.blocks if args.mode == "dirichlet" else None,
               "concentration": args.concentration if args.mode == "dirichlet" else None,
               "labels_used_for_stream_construction_only": True, "sha256": sha256(args.output)})


if __name__ == "__main__":
    main()
