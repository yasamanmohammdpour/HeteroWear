"""
Smoke / property tests on synthetic data (CPU, seconds). No datasets needed.

    python -m unittest discover -s tests -v          # or: pytest tests

What is checked
  * every public model/loss builds and runs forward + backward
  * a missing sensor is *omitted*, never imputed: a window's representation
    must not change when it is batched next to a window with more sensors
    (i.e. padding masks at the cross-sensor and latent-fusion stages work)
  * structured sensor dropout invariants
  * cross-configuration InfoNCE behaves sensibly
  * masked-sensor-modeling targets have the documented per-axis shape
  * the robustness-curve channel-removal masks are deterministic, so that
    HW-Scratch and HW-LODO see identical masks
  * subject-disjoint splits
"""
import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.hw_flat import ChannelInput, HWFlatEncoder
from src.models.heterowear import HeteroWearEncoder
from src.losses.structured_dropout import structured_dropout
from src.losses.consistency import cross_config_consistency_loss
from src.losses.masked_modeling import MSMHead
from src.data.splits import make_subject_split

SMALL = dict(d_model=32, n_patches=8, samples_per_patch=16, temporal_layers=1,
             cross_sensor_layers=1, n_latent_queries=2, latent_layers=1)

# (modality, location, device, native_hz, n_axes)
SENSORS = {
    "wrist_acc": ("ACC", "wrist", "smartwatch", 32.0, 3),
    "wrist_gyro": ("GYRO", "wrist", "smartwatch", 32.0, 3),
    "chest_acc": ("ACC", "chest", "research_imu", 100.0, 3),
    "chest_ecg": ("ECG", "chest", "research_imu", 700.0, 1),
    "wrist_ppg": ("PPG", "wrist", "smartwatch", 64.0, 1),
}


def make_window(names, seed=0, T=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in names:
        mod, loc, dev, hz, ax = SENSORS[n]
        t = T or int(hz * 2)  # 2-second window at native rate -> variable T per sensor
        out.append(ChannelInput(mod, loc, dev, hz, torch.randn(t, ax, generator=g)))
    return out


class TestHeteroWearEncoder(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.enc = HeteroWearEncoder(**SMALL).eval()

    def test_variable_sensor_sets_give_fixed_size_output(self):
        batch = [make_window(["wrist_acc"], 1),
                 make_window(["wrist_acc", "wrist_gyro", "chest_ecg"], 2),
                 make_window(list(SENSORS), 3)]
        g, r, t = self.enc(batch, mask_ratio=0.0)
        self.assertEqual(g.shape, (3, SMALL["d_model"]))
        self.assertIsNone(r) and self.assertIsNone(t)
        self.assertTrue(torch.isfinite(g).all())

    def test_missing_sensors_are_omitted_not_imputed(self):
        small = make_window(["wrist_acc", "chest_ecg"], 5)
        big = make_window(list(SENSORS), 6)
        with torch.no_grad():
            alone, _, _ = self.enc([small], 0.0)
            batched, _, _ = self.enc([small, big], 0.0)
            batched_rev, _, _ = self.enc([big, small], 0.0)
        torch.testing.assert_close(alone[0], batched[0], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(alone[0], batched_rev[1], atol=1e-5, rtol=1e-4)

    def test_sensor_context_changes_representation(self):
        # same signal, different provenance -> different representation
        sig = torch.randn(64, 3)
        a = [ChannelInput("ACC", "wrist", "smartwatch", 32.0, sig)]
        b = [ChannelInput("ACC", "chest", "research_imu", 100.0, sig)]
        with torch.no_grad():
            ga, _, _ = self.enc([a], 0.0)
            gb, _, _ = self.enc([b], 0.0)
        self.assertGreater((ga - gb).abs().max().item(), 1e-4)
        # ...but with context disabled (ablation) the two become identical
        nc = HeteroWearEncoder(**SMALL, use_context=False).eval()
        with torch.no_grad():
            ga, _, _ = nc([a], 0.0)
            gb, _, _ = nc([b], 0.0)
        torch.testing.assert_close(ga, gb, atol=1e-6, rtol=1e-5)

    def test_msm_targets_are_per_axis(self):
        batch = [make_window(["wrist_acc", "chest_ecg"], 7), make_window(["wrist_gyro"], 8)]
        enc = HeteroWearEncoder(**SMALL).train()
        _, reprs, targets = enc(batch, mask_ratio=0.4)
        self.assertEqual(reprs.shape[1], SMALL["d_model"])
        self.assertEqual(targets.shape[1], 6)  # (mean, std) x 3 axes
        self.assertEqual(reprs.shape[0], targets.shape[0])
        self.assertGreater(reprs.shape[0], 0)
        self.assertTrue(torch.isfinite(targets).all())

    def test_one_pretraining_step_backprops_everywhere(self):
        rng = random.Random(0)
        enc = HeteroWearEncoder(**SMALL).train()
        head = MSMHead(SMALL["d_model"])
        base = [make_window(list(SENSORS), s) for s in range(4)]
        va = [structured_dropout(w, rng=rng) for w in base]
        vb = [structured_dropout(w, rng=rng) for w in base]
        g_a, _, _ = enc(va, 0.0)
        g_b, reprs, targets = enc(vb, 0.4)
        loss = cross_config_consistency_loss(g_a, g_b) + \
            torch.nn.functional.mse_loss(head.proj(reprs), targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(enc.latent_fusion.latent_queries.grad)
        self.assertIsNotNone(enc.cross_sensor_encoder.encoder.layers[0].linear1.weight.grad)


class TestStructuredDropout(unittest.TestCase):
    def test_never_empty_and_subset(self):
        w = make_window(list(SENSORS), 0)
        rng = random.Random(1)
        for _ in range(200):
            kept = structured_dropout(w, 0.9, 0.9, 0.9, rng)
            self.assertGreaterEqual(len(kept), 1)
            self.assertTrue(all(any(k is c for c in w) for k in kept))

    def test_zero_probability_keeps_everything(self):
        w = make_window(list(SENSORS), 0)
        self.assertEqual(len(structured_dropout(w, 0, 0, 0, random.Random(0))), len(w))

    def test_whole_modality_removed_together(self):
        w = make_window(["wrist_acc", "chest_acc", "chest_ecg", "wrist_gyro"], 0)  # two ACC streams
        rng = random.Random(3)
        saw_both_gone = False
        for _ in range(300):
            kept = structured_dropout(w, p_modality=0.5, p_location=0.0, p_channel=0.0, rng=rng)
            if len(kept) == 1:
                continue  # every modality was dropped -> one random sensor is restored (by design)
            n_acc = sum(c.modality == "ACC" for c in kept)
            self.assertIn(n_acc, (0, 2))  # a modality is dropped as a whole or not at all
            saw_both_gone |= (n_acc == 0)
        self.assertTrue(saw_both_gone)


class TestLosses(unittest.TestCase):
    def test_consistency_prefers_matched_pairs(self):
        torch.manual_seed(0)
        a = torch.randn(16, 32)
        matched = cross_config_consistency_loss(a, a + 0.01 * torch.randn_like(a))
        shuffled = cross_config_consistency_loss(a, a[torch.randperm(16)])
        self.assertLess(matched.item(), shuffled.item())

    def test_consistency_is_symmetric(self):
        torch.manual_seed(1)
        a, b = torch.randn(8, 16), torch.randn(8, 16)
        torch.testing.assert_close(cross_config_consistency_loss(a, b),
                                   cross_config_consistency_loss(b, a))


class TestFlatBaselineEncoder(unittest.TestCase):
    def test_forward(self):
        enc = HWFlatEncoder(d_model=32, n_layers=1).eval()
        pooled, _, pad, tgt, mm = enc([make_window(["wrist_acc"], 1),
                                       make_window(list(SENSORS), 2)], mask_ratio=0.4)
        self.assertEqual(pooled.shape, (2, 32))
        self.assertTrue(mm.any())


class TestEvalHelpers(unittest.TestCase):
    def test_fixed_channel_dropout_is_deterministic(self):
        from scripts.eval.robustness_curve_heterowear import apply_fixed_channel_dropout
        w = make_window(list(SENSORS), 0)
        for rate in (0.25, 0.5, 0.75):
            a = apply_fixed_channel_dropout(w, rate, window_idx=11)
            b = apply_fixed_channel_dropout(w, rate, window_idx=11)
            self.assertEqual([id(x) for x in a], [id(x) for x in b])
            self.assertGreaterEqual(len(a), 1)
            self.assertLess(len(a), len(w))
        self.assertEqual(len(apply_fixed_channel_dropout(w, 0.0, 11)), len(w))

    def test_heterowear_classifier_end_to_end(self):
        from scripts.eval.finetune_heterowear import HeteroWearClassifier
        clf = HeteroWearClassifier(HeteroWearEncoder(**SMALL), n_classes=4)
        logits = clf([make_window(["wrist_acc"], 1), make_window(list(SENSORS), 2)])
        self.assertEqual(logits.shape, (2, 4))
        logits.sum().backward()


class TestSplits(unittest.TestCase):
    def test_subject_disjoint(self):
        subjects = [f"s{i}" for i in range(20)]
        sp = make_subject_split(subjects, seed=0)
        tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
        self.assertFalse(tr & va or tr & te or va & te)
        self.assertEqual(tr | va | te, set(subjects))


if __name__ == "__main__":
    unittest.main()
