"""Functional and numerical regression checks using synthetic inputs only."""
import unittest
import numpy as np
import torch
from fgm import FGM, FGMConfig


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.text = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)

    def test_refresh_uses_only_prior_observations(self):
        model = FGM(self.text, FGMConfig(min_history=4, update_interval=4))
        rows = torch.nn.functional.normalize(torch.randn(5, 8), dim=1)
        for x in rows[:4]:
            model.step(x)
            self.assertEqual(model.age, 0)
        expected_mean = rows[:4].double().mean(0).float()
        model.step(rows[4])
        self.assertEqual(model.age, 4)
        self.assertTrue(torch.equal(model.mu, expected_mean))
        self.assertEqual(model.n, 5)

    def test_frequency_uses_previous_prior(self):
        model = FGM(self.text)
        scores = model.step(self.text[0])
        self.assertTrue(torch.equal(scores[2], scores[3]))
        prior = model.prior.copy()
        scores = model.step(self.text[1])
        expected = scores[2].numpy() - 2 * np.log(4 * prior).astype("float32")
        np.testing.assert_array_equal(scores[3].numpy(), expected)

    def test_geometry_and_frequency_do_not_change_prototypes(self):
        full = FGM(self.text)
        no_correction = FGM(self.text, FGMConfig(gamma=0, geometry_weight=0))
        for x in torch.nn.functional.normalize(torch.randn(20, 8), dim=1):
            full.step(x)
            no_correction.step(x)
            self.assertTrue(torch.equal(full.A, no_correction.A))
            self.assertTrue(torch.equal(full.V, no_correction.V))

    def test_zero_candidate_retains_previous_unit_anchor(self):
        model = FGM(self.text, FGMConfig(alpha=0, confidence_threshold=1))
        previous = model.A.clone()
        outputs = model.step(self.text[0], torch.zeros(4))
        self.assertTrue(torch.equal(previous, model.A))
        self.assertTrue(all(bool(torch.isfinite(x).all()) for x in outputs))

    def test_reset_removes_history(self):
        model = FGM(self.text)
        model.step(self.text[0])
        model.reset()
        fresh = FGM(self.text)
        self.assertEqual(model.n, 0)
        for a, b in zip(model.step(self.text[1]), fresh.step(self.text[1])):
            self.assertTrue(torch.equal(a, b))

    def test_invalid_config_rejected(self):
        for kwargs in [{"h": 0}, {"gamma": -1}, {"update_interval": 0}, {"temperature": float("nan")}, {"frequency_rate": 1}]:
            with self.assertRaises(ValueError):
                FGMConfig(**kwargs)

    def test_disabled_frequency_returns_geometry_without_log(self):
        model = FGM(self.text, FGMConfig(gamma=0))
        model.prior[:] = 0
        scores = model.step(self.text[0])
        self.assertTrue(torch.equal(scores[2], scores[3]))
        self.assertTrue(bool(torch.isfinite(scores[3]).all()))

    def test_frequency_only_uses_baseline_scores(self):
        model = FGM(self.text, FGMConfig(geometry_weight=0))
        for row in self.text:
            scores = model.step(row)
            self.assertTrue(torch.equal(scores[1].float(), scores[2]))

    @unittest.skipUnless(torch.cuda.is_available(), "FP16 cancellation regression requires CUDA")
    def test_default_fp16_cancellation(self):
        text = torch.zeros((4, 512), device="cuda", dtype=torch.float16)
        text[0, 0] = 1
        for j in range(1, 4):
            text[j, 0] = 0.998
            text[j, j] = (1 - 0.998 ** 2) ** 0.5
        text /= text.norm(dim=1, keepdim=True)
        radius = 0.9404296875
        views = torch.zeros((6, 512), device="cuda", dtype=torch.float16)
        views[:, 0] = -radius
        views[:3, 4] = (1 - radius ** 2) ** 0.5
        views[3:, 4] = -(1 - radius ** 2) ** 0.5
        views /= views.norm(dim=1, keepdim=True)
        x, z = views.mean(0), (100.0 * views @ text.T).mean(0)
        guarded = FGM(text)
        legacy = FGM(text, FGMConfig(normalization="legacy"))
        self.assertTrue(all(bool(torch.isfinite(a).all()) for a in guarded.step(x, z)))
        self.assertFalse(all(bool(torch.isfinite(a).all()) for a in legacy.step(x, z)))


if __name__ == "__main__":
    unittest.main()
