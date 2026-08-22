import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.TracknetV2 import TrackentV2
from utils.TracknetV2.train_val import validate


class _ModeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "train"

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, inputs):
        logits = torch.full(
            (inputs.size(0), 3, 4, 4), -2.0, device=inputs.device
        )
        return torch.sigmoid(logits) if self.mode == "inference" else logits


class _Writer:
    def __init__(self):
        self.values = {}

    def add_scalar(self, tag, value, epoch):
        self.values[tag] = (value, epoch)


class TracknetV2ValidationTests(unittest.TestCase):
    def test_output_mode_controls_activation(self):
        model = TrackentV2(mode="train").eval()
        inputs = torch.randn(1, 9, 16, 16)

        logits = model(inputs)
        model.set_mode("inference")
        probabilities = model(inputs)

        self.assertTrue(torch.allclose(probabilities, torch.sigmoid(logits)))
        with self.assertRaises(ValueError):
            model.set_mode("invalid")

    def test_validate_counts_all_channels_with_temporal_label_alignment(self):
        inputs = torch.zeros(4, 9, 4, 4)
        targets = torch.zeros(4, 3, 4, 4)
        # DataLoader's default collation: labels are preprev, prev, current.
        labels = [
            (torch.zeros(4), torch.zeros(4), torch.tensor([1, 1, 0, 0])),
            (torch.zeros(4), torch.zeros(4), torch.tensor([0, 1, 1, 0])),
            (torch.zeros(4), torch.zeros(4), torch.tensor([0, 0, 1, 1])),
        ]
        centers = [
            (0, 0), None, (1, 1),
            (20, 0), None, None,
            None, (0, 0), (0, 0),
            (0, 0), None, None,
        ]
        writer = _Writer()
        seen_logits = []

        def criterion(logits, _targets):
            seen_logits.append(logits)
            return logits.mean()

        with patch("utils.TracknetV2.train_val.postprocessing", side_effect=centers):
            loss, precision, recall, f1 = validate(
                _ModeModel(),
                [(inputs, targets, labels)],
                criterion,
                "cpu",
                writer=writer,
                epoch=3,
                max_distance=10,
            )

        self.assertLess(float(seen_logits[0].max()), 0.0)
        self.assertAlmostEqual(loss, -2.0)
        self.assertAlmostEqual(precision, 3 / (3 + 3 + 1e-6))
        self.assertAlmostEqual(recall, 3 / (3 + 2 + 1e-6))
        self.assertAlmostEqual(f1, 2 * precision * recall / (precision + recall + 1e-6))
        self.assertEqual(writer.values["Metrics/TP"], (3, 3))
        self.assertEqual(writer.values["Metrics/TN"], (4, 3))
        self.assertEqual(writer.values["Metrics/FP"], (3, 3))
        self.assertEqual(writer.values["Metrics/FN"], (2, 3))
        self.assertAlmostEqual(writer.values["Metrics/Accuracy"][0], 7 / (12 + 1e-6))


if __name__ == "__main__":
    unittest.main()
