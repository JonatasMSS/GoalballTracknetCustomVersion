from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import cv2
import numpy as np
from torch import nn

import trainV2


class TrainV2Tests(unittest.TestCase):
    def assert_paths_rejected(self, root, train_path, val_path, message, extra_args=()):
        logs_path = Path(root) / "logs"
        exps_path = Path(root) / "exps"
        logs_existed, exps_existed = logs_path.exists(), exps_path.exists()
        argv = [
            "trainV2.py", "--train_dataset_path", str(train_path),
            "--val_dataset_path", str(val_path), "--logs_path", str(logs_path),
            "--exps_path", str(exps_path), "--model_path", "weights.pth", *extra_args,
        ]
        stderr = io.StringIO()
        with (
            patch("sys.argv", argv),
            patch("trainV2.os.makedirs") as makedirs,
            patch("trainV2.TrackentV2") as model,
            patch("trainV2.torch.load") as load,
            patch("trainV2.SummaryWriter") as writer,
            patch("trainV2.torch.optim.AdamW") as optimizer,
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as error:
                trainV2.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, stderr.getvalue())
        for operation in (makedirs, model, load, writer, optimizer):
            operation.assert_not_called()
        self.assertEqual(logs_path.exists(), logs_existed)
        self.assertEqual(exps_path.exists(), exps_existed)

    def run_training(self, root, num_epochs, interval=2, f1_scores=None):
        train_path, val_path = Path(root) / "train", Path(root) / "val"
        train_path.mkdir()
        val_path.mkdir()
        argv = [
            "trainV2.py", "--train_dataset_path", str(train_path),
            "--val_dataset_path", str(val_path), "--logs_path", str(Path(root) / "logs"),
            "--exps_path", str(Path(root) / "exps"), "--num_epochs", str(num_epochs),
            "--val_intervals", str(interval),
        ]
        scores = iter([0.0] * num_epochs if f1_scores is None else f1_scores)
        saved = []
        with (
            patch("sys.argv", argv),
            patch("trainV2.torch.cuda.is_available", return_value=False),
            patch("trainV2.os.makedirs"),
            patch("trainV2.TrackentV2", return_value=nn.Linear(1, 1)),
            patch("trainV2.TracknetV2Dataset", side_effect=[[object()], [object()]]),
            patch("trainV2.DataLoader"),
            patch("trainV2.SummaryWriter") as writer,
            patch("trainV2.torch.optim.AdamW"),
            patch("trainV2.torch.optim.lr_scheduler.StepLR"),
            patch("trainV2.train", return_value=0.1) as train,
            patch("trainV2.validate", side_effect=lambda *args, **kwargs: (0.1, 0.2, 0.3, next(scores))) as validate,
            patch("trainV2.torch.save", side_effect=lambda state, path: saved.append((train.call_count, Path(path).name, state))),
            redirect_stdout(io.StringIO()),
        ):
            trainV2.main()
        self.assertEqual(train.call_count, num_epochs)
        writer.return_value.close.assert_called_once()
        return [entry.kwargs["epoch"] + 1 for entry in validate.call_args_list], saved

    def test_main_rejects_identical_dataset_directories(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root) / "dataset"
            dataset_path.mkdir()
            self.assert_paths_rejected(root, dataset_path, dataset_path, "different directories")

    def test_main_rejects_relative_and_absolute_dataset_aliases(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root) / "dataset"
            dataset_path.mkdir()
            self.assert_paths_rejected(
                root, os.path.relpath(dataset_path), dataset_path.resolve(), "different directories"
            )

    @unittest.skipUnless(os.name == "nt", "Case-insensitive directory aliases are Windows-specific")
    def test_main_rejects_windows_case_alias(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root) / "dataset"
            dataset_path.mkdir()
            self.assert_paths_rejected(
                root, dataset_path, str(dataset_path).swapcase(), "different directories"
            )

    def test_main_rejects_symlink_to_same_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            dataset_path = Path(root) / "dataset"
            dataset_path.mkdir()
            alias = Path(root) / "alias"
            try:
                alias.symlink_to(dataset_path, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Directory symlink unavailable: {error}")
            self.assert_paths_rejected(root, dataset_path, alias, "different directories")

    def test_main_rejects_missing_and_file_dataset_paths(self):
        for invalid_option in ("train", "val"):
            for kind in ("missing", "file"):
                with self.subTest(option=invalid_option, kind=kind), tempfile.TemporaryDirectory() as root:
                    paths = {name: Path(root) / name for name in ("train", "val")}
                    paths["val" if invalid_option == "train" else "train"].mkdir()
                    if kind == "file":
                        paths[invalid_option].write_bytes(b"dataset path is a file")
                    self.assert_paths_rejected(
                        root, paths["train"], paths["val"],
                        f"--{invalid_option}_dataset_path must be an existing directory",
                    )

    def test_main_keeps_validation_interval_guard(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, val_path = Path(root) / "train", Path(root) / "val"
            train_path.mkdir()
            val_path.mkdir()
            self.assert_paths_rejected(
                root, train_path, val_path, "val_intervals must be greater than zero",
                extra_args=("--val_intervals", "0"),
            )

    def test_main_rejects_nonpositive_epoch_counts(self):
        for num_epochs in (0, -1):
            with self.subTest(num_epochs=num_epochs), tempfile.TemporaryDirectory() as root:
                train_path, val_path = Path(root) / "train", Path(root) / "val"
                train_path.mkdir()
                val_path.mkdir()
                self.assert_paths_rejected(
                    root, train_path, val_path, "num_epochs must be greater than zero",
                    extra_args=("--num_epochs", str(num_epochs)),
                )

    def test_main_rejects_empty_training_and_validation_datasets_before_side_effects(self):
        for empty_split in ("train", "val"):
            with self.subTest(empty_split=empty_split), tempfile.TemporaryDirectory() as root:
                paths = {name: Path(root) / name for name in ("train", "val")}
                for split, path in paths.items():
                    for directory in ("frames_out", "gts", "labels"):
                        (path / directory).mkdir(parents=True)
                    if split == empty_split:
                        continue
                    for directory in ("frames_out", "gts", "labels"):
                        (path / directory / "match" / "clip").mkdir(parents=True)
                    for index in range(3):
                        filename = f"frame{index}.png"
                        image = np.zeros((360, 640), dtype=np.uint8)
                        for directory in ("frames_out", "gts"):
                            self.assertTrue(cv2.imwrite(str(path / directory / "match" / "clip" / filename), image))
                        (path / "labels" / "match" / "clip" / f"frame{index}.txt").write_text("1 1 1\n")
                self.assert_paths_rejected(
                    root, paths["train"], paths["val"], f"--{empty_split}_dataset_path contains no temporal samples",
                    extra_args=("--original_width", "640", "--original_height", "360"),
                )

    def test_main_preserves_existing_checkpoints(self):
        for filename in ("best_model.pth", "last_model.pth"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                train_path, val_path = Path(root) / "train", Path(root) / "val"
                train_path.mkdir()
                val_path.mkdir()
                checkpoint = Path(root) / "exps" / filename
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"existing checkpoint sentinel")
                self.assert_paths_rejected(root, train_path, val_path, "Use a new --exps_path")
                self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint sentinel")

    def test_main_rejects_dangling_checkpoint_symlink(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, val_path = Path(root) / "train", Path(root) / "val"
            train_path.mkdir()
            val_path.mkdir()
            checkpoint = Path(root) / "exps" / "best_model.pth"
            checkpoint.parent.mkdir()
            target = Path(root) / "missing.pth"
            try:
                checkpoint.symlink_to(target)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"Checkpoint symlink unavailable: {error}")
            self.assert_paths_rejected(root, train_path, val_path, "Use a new --exps_path")
            self.assertTrue(checkpoint.is_symlink())
            self.assertEqual(checkpoint.readlink(), target)
            self.assertFalse(target.exists())

    def test_main_validates_on_intervals_and_final_epoch(self):
        for num_epochs, expected in ((1, [1]), (2, [2]), (3, [2, 3]), (5, [2, 4, 5])):
            with self.subTest(num_epochs=num_epochs), tempfile.TemporaryDirectory() as root:
                epochs, saved = self.run_training(root, num_epochs)
                self.assertEqual(epochs, expected)
                self.assertEqual([epoch for epoch, name, _ in saved if name == "best_model.pth"], [expected[0]])
                self.assertEqual([epoch for epoch, name, _ in saved if name == "last_model.pth"], list(range(1, num_epochs + 1)))

    def test_main_saves_first_zero_f1_and_only_improvements(self):
        with tempfile.TemporaryDirectory() as root:
            epochs, saved = self.run_training(root, 10, f1_scores=[0.0, 0.0, 0.4, 0.3, 0.5])
        self.assertEqual(epochs, [2, 4, 6, 8, 10])
        self.assertEqual([epoch for epoch, name, _ in saved if name == "best_model.pth"], [2, 6, 10])
        self.assertEqual([epoch for epoch, name, _ in saved if name == "last_model.pth"], list(range(1, 11)))
        for _, _, state in saved:
            self.assertEqual(set(state), {"weight", "bias"})

    def test_main_rejects_nonfinite_validation_f1(self):
        for f1 in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(f1=f1), tempfile.TemporaryDirectory() as root:
                with self.assertRaisesRegex(ValueError, "Validation F1 must be finite"):
                    self.run_training(root, 1, f1_scores=[f1])

    def test_main_accepts_distinct_dataset_directories(self):
        with tempfile.TemporaryDirectory() as root:
            train_path, val_path = Path(root) / "train", Path(root) / "val"
            train_path.mkdir()
            val_path.mkdir()
            logs_path, exps_path = Path(root) / "logs", Path(root) / "exps"
            argv = [
                "trainV2.py", "--train_dataset_path", str(train_path),
                "--val_dataset_path", str(val_path), "--logs_path", str(logs_path),
                "--exps_path", str(exps_path),
            ]
            with (
                patch("sys.argv", argv),
                patch("trainV2.torch.cuda.is_available", return_value=False),
                patch("trainV2.os.makedirs") as makedirs,
                patch("trainV2.TracknetV2Dataset", side_effect=[[object()], [object()]]),
                patch("trainV2.TrackentV2", side_effect=RuntimeError("model sentinel")) as model,
                patch("trainV2.torch.load") as load,
                patch("trainV2.SummaryWriter") as writer,
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "model sentinel"):
                    trainV2.main()
            model.assert_called_once_with()
            self.assertEqual(makedirs.call_args_list, [call(str(logs_path)), call(str(exps_path))])
            load.assert_not_called()
            writer.assert_not_called()
            self.assertFalse(logs_path.exists())
            self.assertFalse(exps_path.exists())


if __name__ == "__main__":
    unittest.main()
