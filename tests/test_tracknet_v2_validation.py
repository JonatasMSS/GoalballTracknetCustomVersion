from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.TracknetV2 import TrackentV2
from utils.TracknetV2.train_val import train, validate


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


class _LossModel(_ModeModel):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        logits = inputs[:, :3] + self.bias
        return torch.sigmoid(logits) if self.mode == "inference" else logits


class TracknetV2ValidationTests(unittest.TestCase):
    def test_validate_uses_one_forward_and_exact_logits_and_probabilities(self):
        samples = [
            (torch.full((9, 4, 4), value), torch.zeros(3, 4, 4), [(0.0, 0.0, 0)] * 3)
            for value in (-2.0, 0.0, 2.0)
        ]
        loader = DataLoader(samples, batch_size=2, shuffle=False)
        model = _LossModel()
        criterion = nn.MSELoss(reduction="mean")
        with (
            patch.object(model, "forward", wraps=model.forward) as forward,
            patch.object(criterion, "forward", wraps=criterion.forward) as criterion_forward,
            patch("utils.TracknetV2.train_val.postprocessing", return_value=None) as postprocess,
            redirect_stdout(io.StringIO()),
        ):
            loss, precision, recall, f1 = validate(model, loader, criterion, "cpu")
        self.assertEqual(forward.call_count, len(loader))
        self.assertEqual(criterion_forward.call_count, len(loader))
        self.assertEqual(postprocess.call_count, len(samples) * 3)
        expected_predictions = []
        for forward_call, criterion_call in zip(forward.call_args_list, criterion_forward.call_args_list):
            expected_logits = forward_call.args[0][:, :3] + model.bias.detach()
            logits = criterion_call.args[0]
            torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
            self.assertFalse(logits.requires_grad)
            for probabilities in torch.sigmoid(logits):
                expected_predictions.extend(channel.unsqueeze(0) for channel in probabilities)
        for postprocess_call, expected in zip(postprocess.call_args_list, expected_predictions):
            torch.testing.assert_close(postprocess_call.args[0], expected, rtol=0, atol=0)
        self.assertAlmostEqual(loss, 8 / 3, delta=1e-6)
        self.assertEqual((precision, recall, f1), (0.0, 0.0, 0.0))
        self.assertFalse(model.training)
        self.assertEqual(model.mode, "inference")

    def test_train_validate_train_restores_modes_and_gradients(self):
        samples = [
            (torch.ones(9, 4, 4), torch.zeros(3, 4, 4), [(0.0, 0.0, 0)] * 3)
            for _ in range(2)
        ]
        loader = DataLoader(samples, batch_size=1)
        model = _LossModel().eval()
        model.set_mode("inference")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        criterion = nn.MSELoss(reduction="mean")
        with (
            patch.object(model, "forward", wraps=model.forward) as forward,
            patch("utils.TracknetV2.train_val.get_circle_center_hough", return_value=None),
            redirect_stdout(io.StringIO()),
        ):
            first_loss = train(model, loader, criterion, optimizer, "cpu")
            self.assertEqual(forward.call_count, len(loader))
            self.assertTrue(model.training)
            self.assertEqual(model.mode, "train")
            self.assertIsNotNone(model.bias.grad)
            first_gradient = model.bias.grad.clone()
            forward.reset_mock()

            validation_loss, _, _, _ = validate(model, loader, criterion, "cpu")
            self.assertEqual(forward.call_count, len(loader))
            self.assertFalse(model.training)
            self.assertEqual(model.mode, "inference")
            torch.testing.assert_close(model.bias.grad, first_gradient)
            model.bias.grad = None
            forward.reset_mock()

            second_loss = train(model, loader, criterion, optimizer, "cpu")
            self.assertEqual(forward.call_count, len(loader))
            self.assertTrue(model.training)
            self.assertEqual(model.mode, "train")
            self.assertIsNotNone(model.bias.grad)
            torch.testing.assert_close(model.bias.grad, first_gradient)
        for loss in (first_loss, validation_loss, second_loss):
            self.assertAlmostEqual(loss, 1.0, delta=1e-6)

    def test_epoch_loss_is_sample_mean_for_different_batch_sizes(self):
        samples = [
            (torch.full((9, 4, 4), value), torch.zeros(3, 4, 4), [(0.0, 0.0, 0)] * 3)
            for value in (1.0, 1.0, 1.0, 1.0, 3.0)
        ]
        criterion = nn.MSELoss(reduction="mean")
        for batch_size in (1, 2, 3, 4, 5):
            with self.subTest(batch_size=batch_size):
                loader = DataLoader(samples, batch_size=batch_size, shuffle=False)
                model = _LossModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
                with (
                    patch("utils.TracknetV2.train_val.get_circle_center_hough", return_value=None),
                    redirect_stdout(io.StringIO()),
                ):
                    train_loss = train(model, loader, criterion, optimizer, "cpu")
                    validation_loss, _, _, _ = validate(model, loader, criterion, "cpu")
                self.assertAlmostEqual(train_loss, 2.6, delta=1e-6)
                self.assertAlmostEqual(validation_loss, 2.6, delta=1e-6)
                self.assertIsNotNone(model.bias.grad)

    def test_empty_loaders_raise_in_train_and_validation(self):
        loader = DataLoader([], batch_size=2)
        model = _LossModel()
        criterion = nn.MSELoss(reduction="mean")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        with self.assertRaisesRegex(ValueError, "Training loader contains no samples"):
            train(model, loader, criterion, optimizer, "cpu")
        with self.assertRaisesRegex(ValueError, "Validation loader contains no samples"):
            validate(model, loader, criterion, "cpu")

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
