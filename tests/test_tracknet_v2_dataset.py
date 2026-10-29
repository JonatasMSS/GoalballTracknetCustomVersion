from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from datasets.TracknetV2Dataset import TracknetV2Dataset
from predictV2 import build_input


class TracknetV2DatasetTests(unittest.TestCase):
    def open_dataset(self, root, **overrides):
        root = Path(root)
        options = dict(
            frames_path=str(root / "frames_out"),
            gts_path=str(root / "gts"),
            labels_path=str(root / "labels"),
            original_width=32,
            original_height=16,
            input_width=16,
            input_height=8,
        )
        options.update(overrides)
        return TracknetV2Dataset(**options)

    def make_dataset(self, root, extension=".png"):
        root = Path(root)
        for directory in ("frames_out", "gts", "labels"):
            (root / directory / "match" / "clip").mkdir(parents=True)
        for index in range(4):
            filename = f"frame{index:03d}{extension}"
            frame = np.full((8, 16, 3), 30 * index, dtype=np.uint8)
            heatmap = np.full((8, 16), 60 * index, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(root / "frames_out" / "match" / "clip" / filename), frame))
            self.assertTrue(cv2.imwrite(str(root / "gts" / "match" / "clip" / filename), heatmap))
            label_path = root / "labels" / "match" / "clip" / f"frame{index:03d}.txt"
            label_path.write_text(f"{8 * index} {4 * index} {index}\n", encoding="utf-8")
        dataset = self.open_dataset(root)
        return dataset, root / "labels" / "match" / "clip" / "frame002.txt"

    def test_invalid_original_and_input_dimensions_raise(self):
        invalid_dimensions = {
            "original_width": (0, -1, float("nan"), float("inf")),
            "original_height": (0, -1, float("nan"), float("inf")),
            "input_width": (0, -8, 7, 17, float("nan"), float("inf")),
            "input_height": (0, -8, 7, 17, float("nan"), float("inf")),
        }
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            for option, values in invalid_dimensions.items():
                for value in values:
                    with self.subTest(option=option, value=value):
                        with self.assertRaisesRegex(ValueError, option.split("_")[0]):
                            self.open_dataset(root, **{option: value})

    def test_natural_frame_order_allows_gaps_and_breaks_ties_by_name(self):
        for filenames in (
            ["frame1", "frame2", "frame10", "frame20"],
            ["frame1", "frame02", "frame2", "frame10"],
        ):
            with self.subTest(filenames=filenames), tempfile.TemporaryDirectory() as root:
                self.make_dataset(root)
                for index, stem in enumerate(filenames):
                    for directory, extension in (("frames_out", ".png"), ("gts", ".png"), ("labels", ".txt")):
                        folder = Path(root) / directory / "match" / "clip"
                        (folder / f"frame{index:03d}{extension}").rename(folder / f"{stem}{extension}")
                dataset = self.open_dataset(root)
                self.assertEqual(len(dataset), 2)
                for index in range(len(dataset)):
                    inputs, targets, labels = dataset[index]
                    self.assertEqual(labels, [(4.0 * i, 2.0 * i, i) for i in range(index, index + 3)])
                    for channel in range(3):
                        self.assertAlmostEqual(targets[channel].mean().item(), 60 * (index + channel) / 255, delta=1e-6)

    def test_clips_do_not_share_windows_and_short_clips_need_no_gt(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            short_clip = Path(root) / "frames_out" / "match" / "short"
            short_clip.mkdir()
            for index in range(2):
                (short_clip / f"frame{index}.png").write_bytes(b"not decoded without a temporal window")
            dataset = self.open_dataset(root)
            self.assertEqual(len(dataset), 2)
            for sample in dataset.samples:
                self.assertTrue(all(Path(path).parent.name == "clip" for path in sample[:3]))

    def test_gt_falls_back_to_unique_matching_stem(self):
        for frame_extension, gt_extension in ((".jpg", ".png"), (".png", ".jpg"), (".jpg", ".jpeg")):
            with self.subTest(frame=frame_extension, gt=gt_extension), tempfile.TemporaryDirectory() as root:
                self.make_dataset(root, frame_extension)
                folder = Path(root) / "gts" / "match" / "clip"
                for path in list(folder.iterdir()):
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    self.assertTrue(cv2.imwrite(str(path.with_suffix(gt_extension)), image))
                    path.unlink()
                dataset = self.open_dataset(root)
                self.assertEqual(dataset[0][1].shape, (3, 8, 16))
                self.assertTrue(all(Path(path).suffix == gt_extension for path in dataset.samples[0][3:6]))

    def test_exact_gt_name_wins_over_other_extensions(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            folder = Path(root) / "gts" / "match" / "clip"
            for extension in (".jpg", ".jpeg"):
                self.assertTrue(cv2.imwrite(str(folder / f"frame002{extension}"), np.zeros((8, 16), dtype=np.uint8)))
            dataset = self.open_dataset(root)
            self.assertEqual(Path(dataset.samples[0][3]).name, "frame002.png")
            self.assertAlmostEqual(dataset[0][1][2].mean().item(), 120 / 255, delta=1e-6)

    def test_missing_gt_reports_frame_and_expected_path(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root)
            gt_path = Path(root) / "gts" / "match" / "clip" / "frame002.png"
            gt_path.unlink()
            with self.assertRaises(FileNotFoundError) as error:
                self.open_dataset(root)
            self.assertIn(str(gt_path), str(error.exception))
            self.assertIn(str(Path(root) / "frames_out" / "match" / "clip" / "frame002.png"), str(error.exception))

    def test_ambiguous_gt_reports_frame_and_candidates(self):
        with tempfile.TemporaryDirectory() as root:
            self.make_dataset(root, ".jpg")
            folder = Path(root) / "gts" / "match" / "clip"
            (folder / "frame002.jpg").unlink()
            candidates = [folder / "frame002.png", folder / "frame002.jpeg"]
            for path in candidates:
                self.assertTrue(cv2.imwrite(str(path), np.zeros((8, 16), dtype=np.uint8)))
            with self.assertRaises(ValueError) as error:
                self.open_dataset(root)
            for path in candidates + [Path(root) / "frames_out" / "match" / "clip" / "frame002.jpg"]:
                self.assertIn(str(path), str(error.exception))

    def test_corrupt_frames_and_gt_are_rejected_when_sample_is_read(self):
        for directory, message in (("frames_out", "Unable to read frame"), ("gts", "Unable to read ground truth")):
            for index in range(3):
                with self.subTest(directory=directory, index=index), tempfile.TemporaryDirectory() as root:
                    self.make_dataset(root)
                    path = Path(root) / directory / "match" / "clip" / f"frame{index:03d}.png"
                    path.write_bytes(b"invalid image contents")
                    dataset = self.open_dataset(root)
                    with self.assertRaisesRegex(ValueError, message) as error:
                        dataset[0]
                    self.assertIn(str(path), str(error.exception))

    def test_gt_size_mismatch_is_rejected_without_resize(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, _ = self.make_dataset(root)
            gt_path = Path(root) / "gts" / "match" / "clip" / "frame002.png"
            self.assertTrue(cv2.imwrite(str(gt_path), np.zeros((16, 32), dtype=np.uint8)))
            with self.assertRaisesRegex(ValueError, "expected \\(8, 16\\)") as error:
                dataset[0]
            self.assertIn(str(gt_path), str(error.exception))

    def test_extensions_keep_label_order_scale_and_visibility(self):
        expected_labels = [(0.0, 0.0, 0), (4.0, 2.0, 1), (8.0, 4.0, 2), (12.0, 6.0, 3)]
        for extension in (".jpg", ".png", ".JPG", ".PNG"):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as root:
                dataset, _ = self.make_dataset(root, extension)
                self.assertEqual(len(dataset), 2)
                for sample_index in range(len(dataset)):
                    inputs, targets, labels = dataset[sample_index]
                    self.assertEqual(tuple(inputs.shape), (9, 8, 16))
                    self.assertEqual(tuple(targets.shape), (3, 8, 16))
                    self.assertEqual(labels, expected_labels[sample_index:sample_index + 3])
                    for channel in range(3):
                        self.assertAlmostEqual(
                            targets[channel].mean().item(), 60 * (sample_index + channel) / 255,
                            delta=1e-6,
                        )

    def test_inference_preprocessing_matches_dataset_on_high_frequency_frames(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, _ = self.make_dataset(root)
            rows, columns = np.indices((16, 32))
            paths = []
            for index in range(3):
                frame = np.stack([
                    (columns * 85 + rows * 31 + index * 47) % 256,
                    ((columns + rows + index) % 2) * 255,
                    (columns * 17 + rows * 73 + index * 61) % 256,
                ], axis=-1).astype(np.uint8)
                path = Path(root) / "frames_out" / "match" / "clip" / f"frame{index:03d}.png"
                self.assertTrue(cv2.imwrite(str(path), frame))
                paths.append(str(path))
            dataset_input = dataset.get_input(paths[2], paths[1], paths[0])
            frames = [cv2.imread(path) for path in paths]
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            inference_input = build_input(frames, (8, 16), "cpu", mean, std)
            self.assertEqual(tuple(inference_input.shape), (1, 9, 8, 16))
            torch.testing.assert_close(inference_input[0], dataset_input, rtol=0, atol=1e-7)

    def test_missing_label_raises_with_path(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, label_path = self.make_dataset(root)
            label_path.unlink()
            with self.assertRaises(FileNotFoundError) as error:
                dataset[0]
            self.assertEqual(error.exception.filename, str(label_path))

    def test_invalid_label_raises_with_path(self):
        invalid_labels = [
            "", " \n", "1 2", "1 2 1 extra", "invalid 2 1", "1 invalid 1",
            "1 2 invalid", "1 2 1.5", "nan 2 1", "1 NaN 1", "inf 2 1", "1 -inf 1",
        ]
        with tempfile.TemporaryDirectory() as root:
            dataset, label_path = self.make_dataset(root)
            for label in invalid_labels:
                with self.subTest(label=label):
                    label_path.write_text(label, encoding="utf-8")
                    with self.assertRaises(ValueError) as error:
                        dataset[0]
                    self.assertIn(str(label_path), str(error.exception))

    def test_only_first_line_is_parsed(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, label_path = self.make_dataset(root)
            label_path.write_text("16\t8 2\nignored second line\n", encoding="utf-8")
            _, _, labels = dataset[0]
            self.assertEqual(labels[2], (8.0, 4.0, 2))


if __name__ == "__main__":
    unittest.main()
