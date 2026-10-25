"""Compute accuracy after adaptation has finished and predictions are sealed."""
import argparse
import json
from pathlib import Path
import numpy as np
from fggm.io import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True, help="Integer labels [N], in original cached-row order")
    p.add_argument("--output", type=Path, help="Optional accuracy JSON")
    args = p.parse_args()
    receipt = json.loads(args.predictions.with_suffix(".json").read_text(encoding="utf-8"))
    if receipt["status"] != "SEALED" or receipt["sha256"] != sha256(args.predictions):
        raise ValueError("Prediction file does not match its receipt")
    labels = np.load(args.labels, allow_pickle=False)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("labels must be nonnegative integer class indices [N]")
    with np.load(args.predictions, allow_pickle=False) as saved:
        predictions, order, branches = saved["predictions"], saved["order"], saved["branches"].tolist()
    if not np.array_equal(np.sort(order), np.arange(len(labels))):
        raise ValueError("Prediction order does not cover the label array exactly")
    if predictions.shape != (len(labels), len(branches)) or not len(labels):
        raise ValueError("Invalid prediction shape")
    if "classes" in receipt and int(labels.max()) >= receipt["classes"]:
        raise ValueError("Label indices exceed the supplied class vocabulary")
    counts = {name: int(np.count_nonzero(predictions[:, j] == labels[order])) for j, name in enumerate(branches)}
    result = {"n": len(labels), "correct": counts,
              "accuracy": {name: 100 * count / len(labels) for name, count in counts.items()},
              "predictions_sha256": receipt["sha256"], "labels_sha256": sha256(args.labels)}
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
