from contextlib import contextmanager, redirect_stderr
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import torch
from torch import nn

import predictV2


class _RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.clone())
        return torch.zeros(1, 3, 16, 16, device=inputs.device)


class PredictV2Tests(unittest.TestCase):
    def test_invalid_cli_values_fail_before_model_or_video_io(self):
        invalid_options = [
            ("--input_width", "0"), ("--input_width", "-8"), ("--input_width", "641"),
            ("--input_height", "0"), ("--input_height", "-8"), ("--input_height", "361"),
            ("--threshold", "-0.1"), ("--threshold", "1.1"), ("--threshold", "nan"),
            ("--threshold", "inf"), ("--threshold", "-inf"), ("--trail_length", "-1"),
        ]
        for option, value in invalid_options:
            with self.subTest(option=option, value=value):
                argv = ["predictV2.py", "--model_path", "weights.pth", "--video_path", "input.mp4", f"{option}={value}"]
                stderr = io.StringIO()
                with (
                    patch("sys.argv", argv),
                    patch("predictV2.TrackentV2") as model,
                    patch("predictV2.torch.load") as load,
                    patch("predictV2.cv2.VideoCapture") as capture,
                    patch("predictV2.cv2.VideoWriter") as writer,
                    patch("predictV2.os.makedirs") as makedirs,
                    redirect_stderr(stderr),
                ):
                    with self.assertRaises(SystemExit) as error:
                        predictV2.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn(option, stderr.getvalue())
                for operation in (model, load, capture, writer, makedirs):
                    operation.assert_not_called()

    def assert_same_video_rejected(self, video_path, video_out):
        original_bytes = Path(video_path).read_bytes()
        stderr = io.StringIO()
        argv = [
            "predictV2.py", "--model_path", "weights.pth",
            "--video_path", str(video_path), "--video_out", str(video_out),
        ]
        with (
            patch("sys.argv", argv),
            patch("predictV2.TrackentV2") as model,
            patch("predictV2.torch.load") as load,
            patch("predictV2.cv2.VideoCapture") as capture,
            patch("predictV2.cv2.VideoWriter") as writer,
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as error:
                predictV2.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--video_path and --video_out", stderr.getvalue())
        self.assertEqual(Path(video_path).read_bytes(), original_bytes)
        self.assertEqual(Path(video_out).read_bytes(), original_bytes)
        for operation in (model, load, capture, writer):
            operation.assert_not_called()

    def test_valid_cli_boundaries_reach_model(self):
        for threshold in ("0", "1"):
            with self.subTest(threshold=threshold):
                argv = [
                    "predictV2.py", "--model_path", "weights.pth", "--video_path", "input.mp4",
                    "--input_width", "8", "--input_height", "8",
                    "--threshold", threshold, "--trail_length", "0",
                ]
                with (
                    patch("sys.argv", argv),
                    patch("predictV2.TrackentV2", side_effect=RuntimeError("model sentinel")) as model,
                    patch("predictV2.torch.load") as load,
                    patch("predictV2.cv2.VideoCapture") as capture,
                    patch("predictV2.cv2.VideoWriter") as writer,
                ):
                    with self.assertRaisesRegex(RuntimeError, "model sentinel"):
                        predictV2.main()
                model.assert_called_once_with(mode="inference")
                for operation in (load, capture, writer):
                    operation.assert_not_called()

    def test_main_rejects_identical_input_and_output(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = Path(root) / "input.mp4"
            video_path.write_bytes(b"original video sentinel")
            self.assert_same_video_rejected(video_path, video_path)

    def test_main_rejects_relative_and_absolute_paths_to_same_video(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = Path(root) / "input.mp4"
            video_path.write_bytes(b"original video sentinel")
            self.assert_same_video_rejected(os.path.relpath(video_path), video_path.resolve())

    @unittest.skipUnless(os.name == "nt", "Case-insensitive path comparison is Windows-specific")
    def test_main_rejects_windows_case_alias(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = Path(root) / "input.mp4"
            video_path.write_bytes(b"original video sentinel")
            self.assert_same_video_rejected(video_path, str(video_path).swapcase())

    def test_main_rejects_hardlink_to_input(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = Path(root) / "input.mp4"
            video_path.write_bytes(b"original video sentinel")
            alias = Path(root) / "alias.mp4"
            try:
                os.link(video_path, alias)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Hardlink unavailable: {error}")
            self.assert_same_video_rejected(video_path, alias)

    def test_main_rejects_symlink_to_input(self):
        with tempfile.TemporaryDirectory() as root:
            video_path = Path(root) / "input.mp4"
            video_path.write_bytes(b"original video sentinel")
            alias = Path(root) / "alias.mp4"
            try:
                alias.symlink_to(video_path)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Symlink unavailable: {error}")
            self.assert_same_video_rejected(video_path, alias)

    @contextmanager
    def video_fixture(self):
        frames = [
            np.full((16, 16, 3), (10 + 40 * i, 20 + 40 * i, 30 + 40 * i), dtype=np.uint8)
            for i in range(4)
        ]
        clean_frames = [frame.copy() for frame in frames]
        capture = Mock()
        capture.isOpened.return_value = True
        properties = {
            cv2.CAP_PROP_FPS: 30,
            cv2.CAP_PROP_FRAME_WIDTH: 16,
            cv2.CAP_PROP_FRAME_HEIGHT: 16,
            cv2.CAP_PROP_FRAME_COUNT: len(frames),
        }
        capture.get.side_effect = properties.__getitem__
        capture.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        writer = Mock()
        writer.isOpened.return_value = True
        written_frames = []
        writer.write.side_effect = lambda frame: written_frames.append(frame.copy())
        model = _RecordingModel()
        argv = [
            "predictV2.py", "--model_path", "weights.pth", "--video_path", "input.mp4",
            "--input_width", "16", "--input_height", "16",
        ]

        with (
            patch("sys.argv", argv),
            patch("predictV2.torch.cuda.is_available", return_value=False),
            patch("predictV2.TrackentV2", return_value=model),
            patch("predictV2.torch.load", return_value={}),
            patch("predictV2.cv2.VideoCapture", return_value=capture),
            patch("predictV2.cv2.VideoWriter", return_value=writer) as writer_factory,
            patch("predictV2.os.makedirs") as makedirs,
            patch(
                "utils.TracknetV2.train_val.get_circle_center_hough",
                side_effect=[(8, 8), None, (6, 6), None],
            ),
            patch("predictV2.tqdm") as progress_factory,
        ):
            yield SimpleNamespace(
                model=model, capture=capture, writer=writer, properties=properties,
                clean_frames=clean_frames, written_frames=written_frames,
                writer_factory=writer_factory, makedirs=makedirs,
                progress_factory=progress_factory,
                progress=progress_factory.return_value.__enter__.return_value,
            )

    def test_main_keeps_temporal_inputs_clean_until_eof_for_any_frame_count(self):
        for frame_count, expected_total in ((0, None), (1, 1), (float("nan"), None), (-1, None), (4, 4)):
            with self.subTest(frame_count=frame_count), self.video_fixture() as video:
                video.properties[cv2.CAP_PROP_FRAME_COUNT] = frame_count
                predictV2.main()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                normalized = [
                    (torch.from_numpy(frame[..., ::-1].copy()).permute(2, 0, 1).float() / 255 - mean) / std
                    for frame in video.clean_frames
                ]
                windows = [(0, 0, 0), (0, 0, 1), (0, 1, 2), (1, 2, 3)]
                self.assertEqual(len(video.model.inputs), len(windows))
                for inputs, indices in zip(video.model.inputs, windows):
                    expected = torch.cat([normalized[i] for i in indices]).unsqueeze(0)
                    self.assertTrue(torch.allclose(inputs, expected))

                self.assertEqual(len(video.written_frames), len(video.clean_frames))
                for output, clean in zip(video.written_frames, video.clean_frames):
                    self.assertFalse(np.array_equal(output, clean))
                    np.testing.assert_array_equal(output[8, 8], [0, 0, 255])
                self.assertEqual(video.capture.read.call_count, 5)
                video.progress_factory.assert_called_once_with(total=expected_total)
                self.assertEqual(video.progress.update.call_args_list, [call(1)] * 4)
                video.progress_factory.return_value.__exit__.assert_called_once()
                video.capture.release.assert_called_once()
                video.writer.release.assert_called_once()

    def test_invalid_fps_uses_fallback_and_valid_fps_is_preserved(self):
        for fps, expected in ((0, 30), (-1, 30), (float("nan"), 30), (float("inf"), 30), (float("-inf"), 30), (24, 24)):
            with self.subTest(fps=fps), self.video_fixture() as video:
                video.properties[cv2.CAP_PROP_FPS] = fps
                predictV2.main()
                self.assertEqual(video.writer_factory.call_args.args[2], expected)
                self.assertEqual(len(video.written_frames), 4)
                video.capture.release.assert_called_once()
                video.writer.release.assert_called_once()

    def test_invalid_video_dimensions_release_capture_before_writer_creation(self):
        for property_id in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            for value in (0, -1, 0.5, float("nan"), float("inf")):
                with self.subTest(property_id=property_id, value=value), self.video_fixture() as video:
                    video.properties[property_id] = value
                    with self.assertRaisesRegex(ValueError, "Invalid video dimensions"):
                        predictV2.main()
                    video.capture.release.assert_called_once()
                    video.writer_factory.assert_not_called()
                    video.writer.release.assert_not_called()
                    video.makedirs.assert_not_called()
                    video.capture.read.assert_not_called()

    def test_input_open_failure_releases_capture(self):
        with self.video_fixture() as video:
            video.capture.isOpened.return_value = False
            with self.assertRaisesRegex(ValueError, "Unable to open video"):
                predictV2.main()
            video.capture.release.assert_called_once()
            video.writer_factory.assert_not_called()
            video.writer.release.assert_not_called()

    def test_output_open_failure_releases_capture_and_writer(self):
        with self.video_fixture() as video:
            video.writer.isOpened.return_value = False
            with self.assertRaisesRegex(ValueError, "Unable to open output video"):
                predictV2.main()
            video.capture.release.assert_called_once()
            video.writer.release.assert_called_once()
            video.capture.read.assert_not_called()
            video.progress_factory.assert_not_called()

    def test_prediction_errors_release_capture_writer_and_progress(self):
        for stage in ("forward", "postprocessing"):
            with self.subTest(stage=stage), self.video_fixture() as video:
                error = RuntimeError(f"{stage} sentinel")
                failure = (
                    patch.object(video.model, "forward", side_effect=error)
                    if stage == "forward"
                    else patch("predictV2.postprocessing", side_effect=error)
                )
                with failure, self.assertRaisesRegex(RuntimeError, f"{stage} sentinel"):
                    predictV2.main()
                video.capture.release.assert_called_once()
                video.writer.release.assert_called_once()
                video.writer.write.assert_not_called()
                video.progress.update.assert_not_called()
                video.progress_factory.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
