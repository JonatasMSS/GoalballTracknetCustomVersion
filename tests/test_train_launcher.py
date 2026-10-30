from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

import train as launcher


class TrainLauncherTests(unittest.TestCase):
    def make_document(self, root, model="tracknet_v2"):
        folder = Path(root) / "config folder with spaces"
        folder.mkdir()
        document = yaml.safe_load(Path(launcher.__file__).with_name("config.yaml").read_text(encoding="utf-8"))
        document["model"] = model
        profile = document["models"][model]
        fields = ("train_path", "val_path") if model == "tracknet_v1" else ("train_dataset_path", "val_dataset_path")
        for field, split in zip(fields, ("train", "val")):
            profile[field] = f"data/{split} samples"
            (folder / profile[field]).mkdir(parents=True)
        profile["exps_path"] = "outputs/new experiment/weights"
        if model == "tracknet_v2":
            profile["logs_path"] = "outputs/new experiment/logs"
        return folder / "training settings.yaml", document

    def write_document(self, path, document):
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    def assert_cli_rejected(self, path, document, message):
        self.write_document(path, document)
        with (
            patch("sys.argv", ["train.py", "--config", str(path)]),
            patch("train.importlib.import_module") as import_module,
            redirect_stderr(io.StringIO()) as stderr,
        ):
            with self.assertRaises(SystemExit) as error:
                launcher.main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, stderr.getvalue())
        import_module.assert_not_called()
        self.assertFalse((path.parent / "outputs").exists())

    def test_relative_paths_with_spaces_are_anchored_to_yaml(self):
        with tempfile.TemporaryDirectory() as root:
            path, document = self.make_document(root)
            self.write_document(path, document)
            model, config = launcher.load_config(path)
            self.assertEqual(model, "tracknet_v2")
            for field in ("train_dataset_path", "val_dataset_path", "exps_path", "logs_path"):
                self.assertEqual(config[field], os.path.abspath(path.parent / document["models"][model][field]))
            self.assertIsNone(config["model_path"])
            self.assertEqual(set(config), set(document["models"][model]))
            self.assertFalse((path.parent / "outputs").exists())

    def test_absolute_paths_and_existing_checkpoint_are_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            path, document = self.make_document(root, "tracknet_v1")
            profile = document["models"]["tracknet_v1"]
            checkpoint = path.parent / "initial weights.pt"
            checkpoint.write_bytes(b"checkpoint sentinel")
            for field in ("train_path", "val_path", "exps_path"):
                profile[field] = str(path.parent / profile[field])
            profile["model_path"] = "initial weights.pt"
            self.write_document(path, document)
            _, config = launcher.load_config(path)
            for field in ("train_path", "val_path", "exps_path"):
                self.assertEqual(config[field], profile[field])
            self.assertEqual(config["model_path"], str(checkpoint))
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint sentinel")

    def test_ignores_inactive_profile_contents_but_rejects_unknown_model_ids(self):
        with tempfile.TemporaryDirectory() as root:
            path, document = self.make_document(root)
            document["models"]["tracknet_v1"] = {"invalid": [None, False, "unused"]}
            self.write_document(path, document)
            self.assertEqual(launcher.load_config(path)[0], "tracknet_v2")
            document["models"]["tracknet_v3"] = {}
            self.assert_cli_rejected(path, document, "only tracknet_v1 and tracknet_v2")

    def test_rejects_invalid_top_level_and_active_profile_schema_before_import(self):
        for mutation, message in (
            (lambda doc: doc.clear(), "exactly 'model' and 'models'"),
            (lambda doc: doc.update(extra=True), "exactly 'model' and 'models'"),
            (lambda doc: doc.update(model="unknown"), "model must be"),
            (lambda doc: doc.update(models=[]), "models must be a mapping"),
            (lambda doc: doc["models"].update(tracknet_v2=[]), "models.tracknet_v2 must be a mapping"),
            (lambda doc: doc["models"].pop("tracknet_v2"), "models.tracknet_v2 must be a mapping"),
            (lambda doc: doc["models"]["tracknet_v2"].pop("lr"), "missing=['lr']"),
            (lambda doc: doc["models"]["tracknet_v2"].update(lrr=0.01), "unknown=['lrr']"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root)
                mutation(document)
                self.assert_cli_rejected(path, document, message)

    def test_rejects_invalid_types_and_ranges_in_active_v2_profile(self):
        invalid = {
            "batch_size": [True, 0, -1, 1.5, "4"],
            "num_epochs": [False, 0, -1],
            "val_intervals": [True, 0],
            "lr": [False, 0, -0.1, "0.0001", float("nan"), float("inf"), 10 ** 400],
            "stepper_gamma": [True, 0, float("-inf")],
            "original_width": [True, 0, 1920.0],
            "original_height": [False, -1],
            "parallel": ["false", 0, None],
            "device": ["gpu", False, None],
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as root:
                    path, document = self.make_document(root)
                    document["models"]["tracknet_v2"][field] = value
                    self.assert_cli_rejected(path, document, field)

    def test_rejects_invalid_v1_steps_and_experiment_names(self):
        for field, value in (
            ("steps_per_epoch", True), ("steps_per_epoch", 0),
            ("exp_id", None), ("exp_id", ""), ("exp_id", "."), ("exp_id", ".."),
            ("exp_id", "../escape"), ("exp_id", "nested/run"), ("exp_id", "nested\\run"), ("exp_id", "C:escape"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root, "tracknet_v1")
                document["models"]["tracknet_v1"][field] = value
                self.assert_cli_rejected(path, document, field)

    def test_rejects_invalid_paths_and_aliases_before_import(self):
        cases = (
            ("train_dataset_path", None), ("val_dataset_path", ""), ("exps_path", []),
            ("logs_path", False), ("model_path", ""), ("model_path", "missing.pt"),
            ("model_path", "data/train samples"), ("train_dataset_path", "missing directory"),
            ("val_dataset_path", "settings file"), ("val_dataset_path", "data/train samples/../train samples"),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root)
                (path.parent / "settings file").write_bytes(b"not a dataset")
                document["models"]["tracknet_v2"][field] = value
                message = "different directories" if field == "val_dataset_path" and value.endswith("../train samples") else field
                self.assert_cli_rejected(path, document, message)

    def test_rejects_output_files_and_file_ancestors_before_creating_logs(self):
        for model, field, relative in (
            ("tracknet_v2", "exps_path", "occupied"),
            ("tracknet_v2", "logs_path", "occupied"),
            ("tracknet_v2", "exps_path", "occupied/subdir"),
            ("tracknet_v1", "exps_path", "occupied/subdir"),
        ):
            with self.subTest(model=model, field=field, relative=relative), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root, model)
                occupied = path.parent / "occupied"
                occupied.write_bytes(b"existing output sentinel")
                document["models"][model][field] = relative
                self.assert_cli_rejected(path, document, "occupied by a file")
                self.assertEqual(occupied.read_bytes(), b"existing output sentinel")

    def test_rejects_v1_effective_experiment_directory_occupied_by_file(self):
        with tempfile.TemporaryDirectory() as root:
            path, document = self.make_document(root, "tracknet_v1")
            profile = document["models"]["tracknet_v1"]
            profile["exps_path"] = "existing experiments"
            occupied = path.parent / profile["exps_path"] / profile["exp_id"]
            occupied.parent.mkdir()
            occupied.write_bytes(b"existing experiment sentinel")
            self.assert_cli_rejected(path, document, "occupied by a file")
            self.assertEqual(occupied.read_bytes(), b"existing experiment sentinel")

    def test_main_imports_only_selected_module_and_passes_native_dictionary(self):
        for model in ("tracknet_v1", "tracknet_v2"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root, model)
                self.write_document(path, document)
                expected_model, expected_config = launcher.load_config(path)
                with (
                    patch("sys.argv", ["train.py", "--config", str(path)]),
                    patch("train.importlib.import_module") as import_module,
                ):
                    launcher.main()
                import_module.assert_called_once_with(f"{expected_model}.train")
                import_module.return_value.run_training.assert_called_once_with(expected_config)
                self.assertFalse((path.parent / "outputs").exists())

    def test_default_config_is_anchored_to_launcher_file(self):
        with (
            patch("sys.argv", ["train.py"]),
            patch("train.load_config", return_value=("tracknet_v2", {"native": "dictionary"})) as load_config,
            patch("train.importlib.import_module"),
        ):
            launcher.main()
        load_config.assert_called_once_with(Path(launcher.__file__).resolve().with_name("config.yaml"))

    def test_cli_reports_expected_training_api_errors_without_hiding_runtime_errors(self):
        for error in (ValueError("CUDA requested but is not available"), FileNotFoundError("missing frames_out")):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as root:
                path, document = self.make_document(root)
                with (
                    patch("sys.argv", ["train.py", "--config", str(path)]),
                    patch("train.importlib.import_module") as import_module,
                ):
                    import_module.return_value.run_training.side_effect = error
                    self.write_document(path, document)
                    with redirect_stderr(io.StringIO()) as stderr:
                        with self.assertRaises(SystemExit) as exit_error:
                            launcher.main()
                    self.assertEqual(exit_error.exception.code, 2)
                    self.assertIn(str(error), stderr.getvalue())
        with (
            patch("sys.argv", ["train.py"]),
            patch("train.load_config", return_value=("tracknet_v2", {})),
            patch("train.importlib.import_module") as import_module,
        ):
            import_module.return_value.run_training.side_effect = RuntimeError("forward sentinel")
            with self.assertRaisesRegex(RuntimeError, "forward sentinel"):
                launcher.main()

    def test_subprocess_import_and_cli_from_other_directory_have_no_training_side_effects(self):
        script = Path(launcher.__file__).resolve()
        with tempfile.TemporaryDirectory() as root:
            path, document = self.make_document(root)
            document["models"]["tracknet_v2"]["lr"] = False
            self.write_document(path, document)
            code = (
                f"import sys; sys.path.insert(0, {str(script.parent)!r}); import train; "
                "assert not any(name == 'torch' or name == 'cv2' or name.startswith('tracknet_v') for name in sys.modules); "
                "print('lightweight import OK')"
            )
            imported = subprocess.run([sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(imported.returncode, 0, imported.stderr)
            self.assertIn("lightweight import OK", imported.stdout)
            help_result = subprocess.run([sys.executable, "-B", str(script), "--help"], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("--config", help_result.stdout)
            invalid = subprocess.run([sys.executable, "-B", str(script), "--config", str(path)], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(invalid.returncode, 2, invalid.stderr)
            self.assertIn("lr must be", invalid.stderr)
            self.assertNotIn("Traceback", invalid.stderr)
            self.assertFalse((path.parent / "outputs").exists())

    def test_invalid_yaml_and_python_tags_are_rejected_by_safe_loader(self):
        for text in ("model: [", "!!python/tuple [1, 2]", "- not a mapping", ""):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as root:
                path = Path(root) / "invalid.yaml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    launcher.load_config(path)


if __name__ == "__main__":
    unittest.main()
