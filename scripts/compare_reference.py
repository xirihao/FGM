"""Optional regression against an external archived FGGM implementation.

Only source code and cached features are read. No benchmark label is used.
The reference directory must contain fggm.py and prototype_core.py.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
from fggm import FGGM, FGGMConfig
from fggm.io import write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-code", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--samples", type=int, help="Omit to check every cached observation")
    p.add_argument("--h", type=float, default=20)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.reference_code.resolve()))
    spec = importlib.util.spec_from_file_location("archived_fggm_reference", args.reference_code / "fggm.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    x = np.load(args.cache / "features.npy", mmap_mode="r", allow_pickle=False)
    text = np.load(args.cache / "text_weights.npy", allow_pickle=False)
    logits_path = args.cache / "logits.npy"
    logits = np.load(logits_path, mmap_mode="r", allow_pickle=False) if logits_path.exists() else None
    n = len(x) if args.samples is None else min(args.samples, len(x))
    if n < 1:
        raise ValueError("samples must be positive")
    T = torch.from_numpy(text).cuda()
    old = reference.FGGM(T, h=args.h)
    legacy = FGGM(T, FGGMConfig(h=args.h, normalization="legacy"))
    guarded = FGGM(T, FGGMConfig(h=args.h, normalization="guarded"))
    matches = {"legacy_scores_bitwise_equal": True, "guarded_scores_bitwise_equal": True,
               "legacy_anchors_bitwise_equal": True, "guarded_anchors_bitwise_equal": True,
               "legacy_frequency_bitwise_equal": True, "guarded_frequency_bitwise_equal": True}
    first_difference = None
    with torch.inference_mode():
        for i in range(n):
            xi = torch.from_numpy(x[i].copy()).cuda()
            zi = torch.from_numpy(logits[i].copy()).cuda() if logits is not None else (100.0 * xi[None] @ T.T)[0]
            values = old.step_with_logits(xi, zi)
            if not all(bool(torch.isfinite(a).all()) for a in values):
                raise FloatingPointError(f"Reference has nonfinite scores at row {i}")
            for name, model in [("legacy", legacy), ("guarded", guarded)]:
                current = model.step(xi, zi)
                equal = all(torch.equal(a, b) for a, b in zip(values, current))
                matches[f"{name}_scores_bitwise_equal"] &= equal
                matches[f"{name}_anchors_bitwise_equal"] &= torch.equal(old.A, model.A)
                matches[f"{name}_frequency_bitwise_equal"] &= np.array_equal(old.frequency.prior, model.prior)
                if not equal and first_difference is None:
                    first_difference = {"row": i, "mode": name,
                                        "max_abs_difference": [float((a.float() - b.float()).abs().max()) for a, b in zip(values, current)]}
            if (i + 1) % 256 == 0 or i + 1 == n:
                print(f"Compared {i + 1}/{n}", flush=True)
    result = {"passed": all(matches.values()), "observations": n, "full_stream": n == len(x),
              "dimension": text.shape[1], "classes": len(text), "h": args.h,
              "checks": matches, "first_difference": first_difference,
              "reference_sha256": {name: sha256(args.reference_code / name) for name in ("fggm.py", "prototype_core.py")},
              "source_sha256": sha256(Path(__file__).resolve().parents[1] / "fggm/model.py")}
    write_json(args.output, result)
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
