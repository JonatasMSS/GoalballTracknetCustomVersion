import unittest

import torch
from torch import nn

from utils.helpers import denormalize, make_prediction_grid, show_grid_batches


class _TemporalModel(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.forward_states = []

    def forward(self, inputs):
        self.forward_states.append((self.training, torch.is_grad_enabled()))
        logits = torch.tensor([-2.0, -1.0, 2.0]).view(1, 3, 1, 1)
        logits = logits.expand(inputs.size(0), 3, inputs.size(2), inputs.size(3)) + self.bias
        return torch.sigmoid(logits) if self.mode == "inference" else logits


class TracknetV2HelpersTests(unittest.TestCase):
    def make_inputs(self, batch_size):
        colors = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        return torch.stack([
            ((colors + index * 0.01 - mean) / std).reshape(9, 1, 1).expand(9, 4, 5)
            for index in range(batch_size)
        ])

    def assert_grid(self, grid, count, lower_values):
        self.assertEqual(tuple(grid.shape), (3, 14, count * 7 + 2))
        for index in range(count):
            left = 2 + index * 7
            expected_rgb = torch.tensor([0.4, 0.5, 0.6]) + index * 0.01
            torch.testing.assert_close(grid[:, 2:6, left:left + 5], expected_rgb.view(3, 1, 1).expand(3, 4, 5))
            expected_lower = torch.full((3, 4, 5), lower_values[index])
            torch.testing.assert_close(grid[:, 8:12, left:left + 5], expected_lower)

    def test_denormalize_returns_copy_without_changing_input(self):
        tensor = torch.zeros(3, 4, 5)
        original = tensor.clone()
        result = denormalize(tensor)
        self.assertNotEqual(result.data_ptr(), tensor.data_ptr())
        self.assertTrue(torch.equal(tensor, original))
        expected = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).expand_as(tensor)
        torch.testing.assert_close(result, expected)

    def test_prediction_grid_preserves_mode_and_uses_central_probability(self):
        for mode in ("train", "inference"):
            for wrapped in (False, True):
                for batch_size in (1, 2, 3):
                    with self.subTest(mode=mode, wrapped=wrapped, batch_size=batch_size):
                        target = _TemporalModel(mode)
                        model = nn.DataParallel(target) if wrapped else target
                        inputs = self.make_inputs(batch_size)
                        original = inputs.clone()
                        grid = make_prediction_grid(model, inputs)
                        probability = torch.sigmoid(torch.tensor(-1.0)).item()
                        self.assert_grid(grid, batch_size, [probability] * batch_size)
                        self.assertTrue(torch.equal(inputs, original))
                        self.assertEqual(target.mode, mode)
                        self.assertFalse(target.training)
                        self.assertEqual(target.forward_states, [(False, False)])

    def test_batch_grid_accepts_pairs_and_triples_and_limits_first_batch(self):
        inputs = self.make_inputs(3)
        targets = torch.zeros(3, 3, 4, 5)
        targets[:, 0] = 0.8
        targets[:, 2] = 0.9
        for index in range(3):
            targets[index, 1] = 0.2 + index * 0.1
        original_inputs, original_targets = inputs.clone(), targets.clone()
        for with_labels in (False, True):
            for limit, expected_count in ((1, 1), (2, 2), (10, 3)):
                with self.subTest(with_labels=with_labels, limit=limit):
                    batch = (inputs, targets, []) if with_labels else (inputs, targets)

                    def batches():
                        yield batch
                        raise AssertionError("Only the first batch should be consumed")

                    grid = show_grid_batches(batches(), batch_size=limit)
                    self.assert_grid(grid, expected_count, [0.2 + index * 0.1 for index in range(expected_count)])
                    self.assertTrue(torch.equal(inputs, original_inputs))
                    self.assertTrue(torch.equal(targets, original_targets))

    def test_empty_inputs_and_invalid_limits_raise_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "Dataloader contains no samples"):
            show_grid_batches([])
        inputs, targets = torch.empty(0, 9, 4, 5), torch.empty(0, 3, 4, 5)
        with self.assertRaisesRegex(ValueError, "Batch contains no samples"):
            show_grid_batches([(inputs, targets, [])])
        model = _TemporalModel("train")
        with self.assertRaisesRegex(ValueError, "Batch contains no samples"):
            make_prediction_grid(model, inputs)
        self.assertEqual(model.forward_states, [])
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "batch_size must be greater than zero"):
                show_grid_batches([], batch_size=limit)


if __name__ == "__main__":
    unittest.main()
