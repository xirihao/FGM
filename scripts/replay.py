"""Adapt to one cached stream. This command does not accept ground-truth labels."""
import argparse
from dataclasses import replace
import platform
from pathlib import Path
import numpy as np
import torch
from fggm import FGGM, FGGMConfig, BRANCHES
from fggm.io import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, help="Directory containing features.npy and text_weights.npy")
    p.add_argument("--features", type=Path)
    p.add_argument("--text", type=Path)
    p.add_argument("--logits", type=Path, help="Mean selected-view logits. Automatically used when present in --cache")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/default.json"))
    p.add_argument("--h", type=float, help="Override prototype update parameter (Caltech101: 50)")
    p.add_argument("--gamma", type=float, help="Override frequency strength (class-concentrated streams: 0)")
    p.add_argument("--normalization", choices=["guarded", "legacy"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--chunk-size", type=int, default=1024, help="Number of feature rows resident on the device")
    p.add_argument("--order-seed", type=int, default=None, help="Independent permutation; omitted keeps stored order")
    p.add_argument("--order-file", type=Path, help="Explicit permutation of cached row indices (.npy)")
    args = p.parse_args()
    if args.cache:
        if args.features or args.text:
            p.error("Use --cache or --features/--text, not both")
        args.features = args.cache / "features.npy"
        args.text = args.cache / "text_weights.npy"
        if args.logits is None and (args.cache / "logits.npy").is_file():
            args.logits = args.cache / "logits.npy"
    if not args.features or not args.text:
        p.error("Provide --cache or both --features and --text")
    if args.logits is None:
        for name in ("config.json", "metadata.json"):
            metadata_path = args.features.parent / name
            if metadata_path.is_file():
                import json
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                if (metadata.get("protocol") == "multiview" or metadata.get("views", 1) > 1
                        or metadata.get("views_per_image", 1) > 1):
                    p.error("This cache is marked multiview. Supply its selected-view mean logits with --logits")
    if args.order_seed is not None and args.order_file:
        p.error("Use only one order option")
    if args.chunk_size < 1:
        p.error("--chunk-size must be positive")
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use a CUDA build of PyTorch; CPU checks need --device cpu --dtype float32")
    if device.type == "cpu" and args.dtype == "float16":
        p.error("Use --dtype float32 for CPU checks")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = FGGMConfig.from_json(args.config)
    overrides = {key: getattr(args, key) for key in ("h", "gamma", "normalization") if getattr(args, key) is not None}
    config = replace(config, **overrides)
    X = np.load(args.features, mmap_mode="r", allow_pickle=False)
    T = np.load(args.text, allow_pickle=False)
    Z = np.load(args.logits, mmap_mode="r", allow_pickle=False) if args.logits else None
    if X.ndim != 2 or T.ndim != 2 or len(X) == 0 or X.shape[1] != T.shape[1]:
        raise ValueError("Expected nonempty features [N,d] and text [C,d]")
    if Z is not None and Z.shape != (len(X), len(T)):
        raise ValueError("logits must have shape [N,C]")
    if args.dtype == "float16" and any(a.dtype != np.float16 for a in (X, T) + ((Z,) if Z is not None else ())):
        raise TypeError("FP16 reproduction requires FP16 cached arrays; no implicit precision conversion")
    if not np.isfinite(T).all():
        raise ValueError("text contains nonfinite values")
    if args.order_file:
        order = np.load(args.order_file, allow_pickle=False)
        if order.dtype.kind not in "iu" or not np.array_equal(np.sort(order), np.arange(len(X))):
            raise ValueError("order-file must be a permutation of all cached rows")
    elif args.order_seed is not None:
        order = np.random.default_rng(args.order_seed).permutation(len(X))
    else:
        order = np.arange(len(X))
    dtype = getattr(torch, args.dtype)
    model = FGGM(torch.from_numpy(T).to(device=device, dtype=dtype), config)
    predictions = np.empty((len(X), len(BRANCHES)), dtype=np.int32)
    with torch.inference_mode():
        for start in range(0, len(X), args.chunk_size):
            ids = order[start:start + args.chunk_size]
            xc = np.asarray(X[ids])
            zc = np.asarray(Z[ids]) if Z is not None else None
            if not np.isfinite(xc).all() or (zc is not None and not np.isfinite(zc).all()):
                raise ValueError(f"Nonfinite inputs in chunk starting at {start}")
            xb = torch.from_numpy(xc).to(device=device, dtype=dtype)
            zb = torch.from_numpy(zc).to(device=device, dtype=dtype) if zc is not None else None
            for j, x in enumerate(xb):
                values = model.step(x, zb[j] if zb is not None else None)
                if not all(bool(torch.isfinite(value).all()) for value in values):
                    raise FloatingPointError(f"Nonfinite scores at observation {start + j + 1}")
                # topk tie handling matches the evaluated implementation.
                predictions[start + j] = [int(value[None].topk(1, 1, True, True).indices[0, 0]) for value in values]
            print(f"Adapted {min(start + args.chunk_size, len(X))}/{len(X)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, predictions=predictions, order=order, branches=np.asarray(BRANCHES))
    inputs = {"features": sha256(args.features), "text": sha256(args.text)}
    if args.logits:
        inputs["logits"] = sha256(args.logits)
    if args.order_file:
        inputs["order"] = sha256(args.order_file)
    receipt = {"status": "SEALED", "n": len(X), "classes": len(T), "dimension": T.shape[1],
               "sha256": sha256(args.output), "inputs_sha256": inputs, "config": config.to_dict(),
               "order_seed": args.order_seed, "explicit_order": bool(args.order_file), "labels_opened": False,
               "runtime": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                           "cuda": torch.version.cuda, "device": str(device), "dtype": args.dtype,
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                           "tf32": False}}
    write_json(args.output.with_suffix(".json"), receipt)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
