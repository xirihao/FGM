"""A complete cache -> adaptation -> scoring smoke test with no external data."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_permuted_replay_and_independent_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            rng = np.random.default_rng(31)
            x = rng.normal(size=(12, 8)).astype("float32")
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            text = rng.normal(size=(4, 8)).astype("float32")
            text /= np.linalg.norm(text, axis=1, keepdims=True)
            np.save(work / "features.npy", x)
            np.save(work / "text_weights.npy", text)
            labels = np.arange(12, dtype=np.int64) % 4
            np.save(work / "labels.npy", labels)
            output = work / "predictions.npz"
            subprocess.run([sys.executable, str(ROOT / "scripts/replay.py"), "--cache", str(work),
                "--config", str(ROOT / "configs/default.json"), "--output", str(output),
                "--device", "cpu", "--dtype", "float32", "--chunk-size", "5", "--order-seed", "11"], check=True, capture_output=True)
            score = work / "accuracy.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/score.py"), "--predictions", str(output),
                "--labels", str(work / "labels.npy"), "--output", str(score)], check=True, capture_output=True)
            with np.load(output) as stored:
                expected = 100 * (stored["predictions"] == labels[stored["order"]][:, None]).mean(0)
                names = stored["branches"].tolist()
            actual = json.loads(score.read_text())["accuracy"]
            np.testing.assert_allclose([actual[k] for k in names], expected)
            receipt = json.loads(output.with_suffix(".json").read_text())
            self.assertFalse(receipt["labels_opened"])
            # The scoring boundary rejects any modified prediction archive.
            with output.open("ab") as stream:
                stream.write(b"modified")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/score.py"), "--predictions", str(output),
                "--labels", str(work / "labels.npy")], capture_output=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
