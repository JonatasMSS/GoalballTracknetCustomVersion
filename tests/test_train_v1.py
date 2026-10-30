from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

from torch import nn

from tracknet_v1 import train as training


class TrainV1Tests(unittest.TestCase):
    def make_config(self, root, **overrides):
        root = Path(root)
        config = {
            "batch_size": 2,
            "exp_id": "default",
            "num_epochs": 500,
            "lr": 1e-4,
            "val_intervals": 2,
            "steps_per_epoch": 200,
            "exps_path": str(root / "runs"),
            "train_path": str(root / "train"),
            "val_path": str(root / "val"),
            "model_path": None,
            "device": "auto",
        }
        config.update(overrides)
        return config

    def assert_rejected(self, config, message):
        with (
            patch("tracknet_v1.train.os.makedirs") as makedirs,
            patch("tracknet_v1.train.BallTrackerNet") as model,
            patch("tracknet_v1.train.torch.load") as load,
            patch("tracknet_v1.train.SummaryWriter") as writer,
            patch("tracknet_v1.train.optim.Adam") as optimizer,
            patch("tracknet_v1.train.torch.utils.data.DataLoader") as loader,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(ValueError, message):
                training.run_training(config)
        for operation in (makedirs, model, load, writer, optimizer, loader):
            operation.assert_not_called()
        self.assertFalse(Path(config["exps_path"]).exists())

    def test_rejects_invalid_or_unavailable_device_before_dataset_access(self):
        for device, message in (("gpu", "device must be"), ("cuda", "CUDA requested but is not available")):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as root:
                with (
                    patch("tracknet_v1.train.torch.cuda.is_available", return_value=False),
                    patch("tracknet_v1.train.TracknetV1Dataset") as dataset,
                ):
                    self.assert_rejected(self.make_config(root, device=device), message)
                dataset.assert_not_called()

    def test_requires_existing_distinct_explicit_split_directories(self):
        for option in ("train_path", "val_path"):
            for kind in ("none", "missing", "file", "same", "relative_alias"):
                with self.subTest(option=option, kind=kind), tempfile.TemporaryDirectory() as root:
                    config = self.make_config(root)
                    Path(config["train_path"]).mkdir()
                    Path(config["val_path"]).mkdir()
                    if kind == "none":
                        config[option] = None
                    elif kind in ("missing", "file"):
                        invalid_path = Path(root) / "invalid"
                        if kind == "file":
                            invalid_path.write_bytes(b"not a directory")
                        config[option] = str(invalid_path)
                    else:
                        other = config["val_path" if option == "train_path" else "train_path"]
                        config[option] = os.path.relpath(other) if kind == "relative_alias" else other
                    message = "different directories" if kind in ("same", "relative_alias") else "must be an existing directory"
                    with patch("tracknet_v1.train.TracknetV1Dataset") as dataset:
                        self.assert_rejected(config, message)
                    dataset.assert_not_called()

    def test_rejects_empty_splits_and_nonpositive_schedule_before_artifacts(self):
        cases = (("train", {}), ("val", {}), (None, {"val_intervals": 0}), (None, {"num_epochs": 0}))
        for empty_split, overrides in cases:
            with self.subTest(empty_split=empty_split, overrides=overrides), tempfile.TemporaryDirectory() as root:
                config = self.make_config(root, **overrides)
                for option in ("train_path", "val_path"):
                    for directory in ("frames_out", "gts", "labels"):
                        (Path(config[option]) / directory).mkdir(parents=True)
                datasets = [[], [object()]] if empty_split == "train" else [[object()], []]
                if empty_split is not None:
                    message = f"{empty_split}_path contains no temporal samples"
                    with patch("tracknet_v1.train.TracknetV1Dataset", side_effect=datasets):
                        self.assert_rejected(config, message)
                else:
                    option = next(iter(overrides))
                    self.assert_rejected(config, f"{option} must be greater than zero")

    def test_selects_requested_or_automatic_device(self):
        for device, available, expected in (
            ("auto", False, "cpu"), ("auto", True, "cuda"),
            ("cpu", True, "cpu"), ("cuda", True, "cuda"),
        ):
            with self.subTest(device=device, available=available), tempfile.TemporaryDirectory() as root:
                config = self.make_config(root, device=device)
                Path(config["train_path"]).mkdir()
                Path(config["val_path"]).mkdir()
                with (
                    patch("tracknet_v1.train.torch.cuda.is_available", return_value=available),
                    patch("tracknet_v1.train.TracknetV1Dataset", side_effect=[[object()], [object()]]),
                    patch("tracknet_v1.train.torch.utils.data.DataLoader"),
                    patch("tracknet_v1.train.BallTrackerNet") as model,
                    patch("tracknet_v1.train.os.makedirs") as makedirs,
                    patch("tracknet_v1.train.SummaryWriter") as writer,
                    redirect_stdout(io.StringIO()),
                ):
                    model.return_value.to.side_effect = RuntimeError("device sentinel")
                    with self.assertRaisesRegex(RuntimeError, "device sentinel"):
                        training.run_training(config)
                model.return_value.to.assert_called_once_with(expected)
                makedirs.assert_not_called()
                writer.assert_not_called()

    def test_native_api_preserves_splits_schedule_optimizer_steps_and_checkpoint_rules(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.make_config(root, num_epochs=10, steps_per_epoch=7, device="cpu", model_path="initial.pt")
            train_path, val_path = Path(config["train_path"]), Path(config["val_path"])
            train_path.mkdir()
            val_path.mkdir()
            datasets = [[object(), object()], [object()]]
            model = nn.Linear(1, 1)
            saved = []
            with (
                patch("tracknet_v1.train.TracknetV1Dataset", side_effect=datasets) as dataset_factory,
                patch("tracknet_v1.train.torch.utils.data.random_split", side_effect=AssertionError("random split forbidden")),
                patch("tracknet_v1.train.torch.utils.data.DataLoader", side_effect=["train loader", "val loader"]) as loader,
                patch("tracknet_v1.train.BallTrackerNet", return_value=model),
                patch("tracknet_v1.train.torch.load", return_value=model.state_dict()) as load,
                patch.object(model, "load_state_dict", wraps=model.load_state_dict) as load_state,
                patch("tracknet_v1.train.os.makedirs") as makedirs,
                patch("tracknet_v1.train.SummaryWriter") as writer,
                patch("tracknet_v1.train.optim.Adam") as optimizer,
                patch("tracknet_v1.train.train", return_value=0.1) as train,
                patch("tracknet_v1.train.validate", side_effect=[(0.2, 0.1, 0.1, f1) for f1 in (0, 0.4, 0.3, 0.6)]) as validate,
                patch("tracknet_v1.train.torch.save", side_effect=lambda state, path: saved.append((train.call_count, Path(path).name, state))),
                redirect_stdout(io.StringIO()),
            ):
                optimizer.return_value.param_groups = [{"lr": config["lr"]}]
                training.run_training(config)
            self.assertEqual(dataset_factory.call_args_list, [
                call(frames_path=str(path / "frames_out"), gts_path=str(path / "gts"), labels_path=str(path / "labels"))
                for path in (train_path, val_path)
            ])
            self.assertEqual(loader.call_args_list, [
                call(datasets[0], batch_size=2, shuffle=True), call(datasets[1], batch_size=2, shuffle=False),
            ])
            self.assertEqual(train.call_count, 10)
            for epoch, invocation in enumerate(train.call_args_list):
                self.assertEqual(invocation.args[1:], ("train loader", optimizer.return_value, "cpu", epoch, 7))
            self.assertEqual([entry.args[3] + 1 for entry in validate.call_args_list], [3, 5, 7, 9])
            self.assertEqual([(epoch, name) for epoch, name, _ in saved], [
                (3, "model_last.pt"), (5, "model_best.pt"), (5, "model_last.pt"),
                (7, "model_last.pt"), (9, "model_best.pt"), (9, "model_last.pt"),
            ])
            for _, _, state in saved:
                self.assertEqual(set(state), {"weight", "bias"})
            self.assertEqual(optimizer.call_args.kwargs, {"lr": config["lr"]})
            load.assert_called_once_with("initial.pt", map_location="cpu")
            load_state.assert_called_once()
            makedirs.assert_called_once_with(str(Path(config["exps_path"]) / "default" / "plots"))
            writer.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
