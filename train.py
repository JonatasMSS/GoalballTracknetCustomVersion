import argparse
import importlib
import math
import os
from pathlib import Path

import yaml


def load_config(path):
    """Read the selected model profile and resolve paths relative to its YAML."""
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {config_path}: {error}") from error
    if not isinstance(document, dict) or set(document) != {"model", "models"}:
        raise ValueError("Configuration must contain exactly 'model' and 'models'")
    model = document["model"]
    model_names = ("tracknet_v1", "tracknet_v2")
    if model not in model_names:
        raise ValueError("model must be 'tracknet_v1' or 'tracknet_v2'")
    profiles = document["models"]
    if not isinstance(profiles, dict) or any(name not in model_names for name in profiles):
        raise ValueError("models must be a mapping containing only tracknet_v1 and tracknet_v2 profiles")
    if not isinstance(profiles.get(model), dict):
        raise ValueError(f"models.{model} must be a mapping")
    config = dict(profiles[model])
    integer_fields = {"batch_size", "num_epochs", "val_intervals"}
    number_fields = {"lr"}
    input_fields = ("train_path", "val_path") if model == "tracknet_v1" else ("train_dataset_path", "val_dataset_path")
    output_fields = ("exps_path",) if model == "tracknet_v1" else ("exps_path", "logs_path")
    extra_fields = {"exp_id", "steps_per_epoch"} if model == "tracknet_v1" else {"stepper_gamma", "original_width", "original_height", "parallel"}
    required = integer_fields | number_fields | set(input_fields) | set(output_fields) | extra_fields | {"model_path", "device"}
    if set(config) != required:
        missing = sorted(required - set(config))
        unknown = sorted(set(config) - required, key=repr)
        raise ValueError(f"Invalid fields in models.{model}: missing={missing}, unknown={unknown}")
    integer_fields |= {"steps_per_epoch"} if model == "tracknet_v1" else {"original_width", "original_height"}
    number_fields |= {"stepper_gamma"} if model == "tracknet_v2" else set()
    for field in integer_fields:
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field in number_fields:
        value = config[field]
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"{field} must be a finite positive number")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    if model == "tracknet_v2" and type(config["parallel"]) is not bool:
        raise ValueError("parallel must be a boolean")
    if model == "tracknet_v1":
        exp_id = config["exp_id"]
        if not isinstance(exp_id, str) or not exp_id.strip() or exp_id in (".", "..") or any(char in exp_id for char in '/\\:'):
            raise ValueError("exp_id must be a single nonempty directory name")

    path_fields = (*input_fields, *output_fields, "model_path")
    for field in path_fields:
        value = config[field]
        if field == "model_path" and value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a nonempty path" + (" or null" if field == "model_path" else ""))
        config[field] = os.path.abspath(config_path.parent / value)
    for field in input_fields:
        if not Path(config[field]).is_dir():
            raise ValueError(f"{field} must be an existing directory: {config[field]}")
    if os.path.samefile(config[input_fields[0]], config[input_fields[1]]):
        raise ValueError(f"{input_fields[0]} and {input_fields[1]} must be different directories")
    if config["model_path"] is not None and not Path(config["model_path"]).is_file():
        raise ValueError(f"model_path must be an existing file: {config['model_path']}")
    output_paths = [(field, Path(config[field])) for field in output_fields]
    if model == "tracknet_v1":
        output_paths.append(("exps_path/exp_id/plots", Path(config["exps_path"]) / config["exp_id"] / "plots"))
    for field, output_path in output_paths:
        for candidate in (output_path, *output_path.parents):
            if (candidate.exists() or candidate.is_symlink()) and not candidate.is_dir():
                raise ValueError(f"{field} must be a directory path; occupied by a file: {candidate}")
    return model, config


def main():
    parser = argparse.ArgumentParser(description="Train TrackNet from a YAML configuration")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().with_name("config.yaml"), help="Path to the training YAML")
    args = parser.parse_args()
    try:
        model, config = load_config(args.config)
        importlib.import_module(f"{model}.train").run_training(config)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
