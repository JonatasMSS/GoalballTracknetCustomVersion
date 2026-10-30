from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tracknet_v1.dataset import TracknetV1Dataset
from tracknet_v1 import infer, train_val


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs[0, :, 0, 0].clone())
        return torch.zeros(inputs.shape[0], 2, 360 * 640, device=inputs.device)


class TracknetV1ContractTests(unittest.TestCase):
    def test_real_dataset_and_inference_agree_on_bgr_temporal_order_and_scale(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for directory in ("frames_out", "gts", "labels"):
                (root / directory / "match" / "clip").mkdir(parents=True)
            for index in range(3):
                name = f"frame{index}"
                frame = np.full((16, 32, 3), (10 + index * 30, 40 + index * 30, 70 + index * 30), dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(root / "frames_out" / "match" / "clip" / f"{name}.jpg"), frame))
                self.assertTrue(cv2.imwrite(str(root / "gts" / "match" / "clip" / f"{name}.png"), np.full((16, 32, 3), 127, dtype=np.uint8)))
                (root / "labels" / "match" / "clip" / f"{name}.txt").write_text("100 200 3\n")
            dataset = TracknetV1Dataset(
                frames_path=str(root / "frames_out"), gts_path=str(root / "gts"), labels_path=str(root / "labels"),
            )
            self.assertEqual(len(dataset), 1)
            inputs, targets, x, y, visibility = dataset[0]
            current, previous, preprevious = [cv2.resize(cv2.imread(path), (640, 360)) for path in dataset.samples[0][:3]]
            expected = np.concatenate((current, previous, preprevious), axis=2).astype(np.float32) / 255
            np.testing.assert_array_equal(inputs, expected.transpose(2, 0, 1))
            self.assertEqual(inputs.shape, (9, 360, 640))
            self.assertEqual(inputs.dtype, np.float32)
            torch.testing.assert_close(infer.get_input_tensor(current, previous, preprevious), torch.from_numpy(inputs).unsqueeze(0), rtol=0, atol=0)
            self.assertEqual(targets.shape, (360 * 640,))
            self.assertEqual(targets.dtype, np.uint8)
            self.assertTrue(np.all(targets == 127))
            self.assertEqual((x, y, visibility), (100, 200, 3))

    def test_postprocessing_preserves_distinct_training_and_inference_calibrations(self):
        circles = np.array([[[100, 50, 3]]], dtype=np.float32)
        for function, arguments, threshold, hough_parameters in (
            (train_val.postprocess, (), 50, {"param1": 47, "param2": 10, "minRadius": 1}),
            (infer.postprocess, (1920, 1080), 127, {"param1": 50, "param2": 2, "minRadius": 2}),
        ):
            with self.subTest(function=function.__module__):
                feature_map = np.zeros(360 * 640, dtype=np.int64)
                feature_map[0] = 1
                with (
                    patch("cv2.threshold", wraps=cv2.threshold) as threshold_call,
                    patch("cv2.HoughCircles", return_value=circles) as hough,
                ):
                    self.assertEqual(function(feature_map, *arguments), (300, 150))
                self.assertEqual(threshold_call.call_args.args[1:], (threshold, 255, cv2.THRESH_BINARY))
                self.assertEqual(hough.call_args.args[0].shape, (360, 640))
                self.assertEqual(hough.call_args.args[0][0, 0], 255)
                self.assertEqual(hough.call_args.kwargs, {"dp": 1, "minDist": 1, "maxRadius": 7, **hough_parameters})

    def test_validation_uses_euclidean_math_distance_and_preserves_metrics(self):
        model = nn.Sequential(nn.Conv2d(9, 2, 1), nn.Flatten(start_dim=2))
        batch = (
            torch.zeros(2, 9, 1, 1), torch.zeros(2, 1, dtype=torch.long),
            torch.tensor([13, 200]), torch.tensor([24, 131]), torch.tensor([1, 3]),
        )
        with (
            patch("tracknet_v1.train_val.postprocess", side_effect=[(10, 20), (200, 100)]),
            patch("tracknet_v1.train_val.math.dist", wraps=math.dist) as distance,
            redirect_stdout(io.StringIO()),
        ):
            loss, precision, recall, f1 = train_val.validate(model, [batch], "cpu", epoch=0)
        self.assertTrue(math.isfinite(loss))
        for value in (precision, recall, f1):
            self.assertAlmostEqual(value, 0.5)
        self.assertEqual(distance.call_count, 2)
        self.assertEqual([tuple(float(value) for value in entry.args[1]) for entry in distance.call_args_list], [(13, 24), (200, 131)])
        self.assertFalse(model.training)

    def test_training_preserves_legacy_max_iters_boundary(self):
        for limit, expected_steps in ((0, 1), (1, 2)):
            with self.subTest(limit=limit):
                model = nn.Sequential(nn.Conv2d(9, 2, 1), nn.Flatten(start_dim=2))
                loader = DataLoader(TensorDataset(torch.ones(5, 9, 1, 1), torch.zeros(5, 1, dtype=torch.long)), batch_size=1)
                optimizer = torch.optim.SGD(model.parameters(), lr=0)
                with (
                    patch.object(optimizer, "step", wraps=optimizer.step) as step,
                    redirect_stdout(io.StringIO()),
                ):
                    loss = train_val.train(model, loader, optimizer, "cpu", epoch=0, max_iters=limit)
                self.assertTrue(math.isfinite(loss))
                self.assertEqual(step.call_count, expected_steps)
                self.assertTrue(model.training)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_module_inference_accepts_one_to_four_videos_and_creates_output_parent(self):
        for count in range(1, 5):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                videos = [root / f"video{index}.mp4" for index in range(count)]
                checkpoint = root / "weights.pt"
                for path in [*videos, checkpoint]:
                    path.write_bytes(b"synthetic sentinel")
                caps = []
                frames = [np.full((16, 32, 3), (10 + 30 * index, 20 + 30 * index, 30 + 30 * index), dtype=np.uint8) for index in range(2)]
                for _ in videos:
                    cap = Mock()
                    cap.isOpened.return_value = True
                    cap.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 32, cv2.CAP_PROP_FRAME_HEIGHT: 16, cv2.CAP_PROP_FPS: 24, cv2.CAP_PROP_FRAME_COUNT: 2}.__getitem__
                    cap.read.side_effect = [(True, frame.copy()) for frame in frames]
                    caps.append(cap)
                model = _RecordingModel()
                argv = ["tracknet_v1.infer", "--video_paths", *map(str, videos), "--model_path", str(checkpoint), "--device", "cpu"]
                output = Path(infer.__file__).resolve().parents[1] / "runs" / "tracknet_v1" / "inference" / "output.mp4"
                if count == 4:
                    output = root / "nested" / "custom.mp4"
                    argv.extend(["--output_path", str(output), "--trail_length", "2", "--start_time", "0", "--max_seconds", "1"])
                with (
                    patch("sys.argv", argv),
                    patch("tracknet_v1.infer.BallTrackerNet", return_value=model),
                    patch("tracknet_v1.infer.torch.load", return_value={}) as load,
                    patch("tracknet_v1.infer.cv2.VideoCapture", side_effect=caps),
                    patch("tracknet_v1.infer.cv2.VideoWriter") as writer,
                    patch("tracknet_v1.infer.os.makedirs") as makedirs,
                    patch("tracknet_v1.infer.postprocess", return_value=(None, None)),
                    patch("tracknet_v1.infer.tqdm", side_effect=lambda iterable, **kwargs: iterable),
                    redirect_stdout(io.StringIO()),
                ):
                    infer.main()
                load.assert_called_once_with(str(checkpoint), map_location="cpu")
                makedirs.assert_called_once_with(str(output.parent), exist_ok=True)
                self.assertEqual(writer.call_args.args[0], str(output))
                self.assertEqual(writer.call_args.args[2:], (24, (32, 16)))
                self.assertEqual(writer.return_value.write.call_count, 2)
                self.assertEqual(len(model.inputs), 2 * count)
                for index, tensor in enumerate(model.inputs):
                    values = [10, 20, 30] * 3 if index < count else [40, 50, 60, 10, 20, 30, 10, 20, 30]
                    torch.testing.assert_close(tensor, torch.tensor(values, dtype=torch.float32) / 255, rtol=0, atol=0)
                self.assertFalse(model.training)
                for cap in caps:
                    cap.release.assert_called_once()
                writer.return_value.release.assert_called_once()

    def test_module_inference_rejects_more_than_four_videos_before_model_loading(self):
        with (
            patch("sys.argv", ["tracknet_v1.infer", "--video_paths", "a", "b", "c", "d", "e", "--model_path", "weights.pt"]),
            patch("tracknet_v1.infer.BallTrackerNet") as model,
            patch("tracknet_v1.infer.os.makedirs") as makedirs,
            redirect_stdout(io.StringIO()),
        ):
            infer.main()
        model.assert_not_called()
        makedirs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
