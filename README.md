# Tracknet Goalball

Python/PyTorch implementations of TrackNet V1 and V2 for ball tracking in goalball videos.
The versions have different input, target and checkpoint contracts; their weights are not interchangeable.

## Environment

Use a virtual environment and install the project dependencies:

```powershell
pip install -r requirements.txt
```

V1 also imports `scipy`, which is not yet listed in `requirements.txt`.
`torchsummary` is optional for V2's `show_model_summary()`; the notebook does not require it.
V1 training selects CUDA explicitly; V2 and video inference select CPU/CUDA according to availability.

## V2 dataset contract

Use separate training and validation roots, each with this hierarchy:

```text
assets/train/
  frames_out/<partida>/<clip>/frame1.jpg
  gts/<partida>/<clip>/frame1.png
  labels/<partida>/<clip>/frame1.txt
assets/val/
  ... same hierarchy, independent matches/clips ...
```

- Each label's first line must contain exactly `x y vis`: finite pixel coordinates in the original resolution and integer visibility. Set `--original_width`/`--original_height` to that resolution (defaults: 1920/1080). Missing, empty or malformed labels raise errors. Visibility `0` must be explicitly annotated; evaluation treats only `vis == 1` as visible.
- GTs are grayscale heatmaps already at the model input size (default: height 360, width 640). They are not resized automatically. The exact frame filename is preferred; otherwise a unique JPG/JPEG/PNG with the same stem is accepted. Missing or ambiguous matches raise errors.
- Frames use natural numeric ordering (`frame1`, `frame2`, `frame10`), with filename tie-breaking. Windows stay inside each clip; numeric gaps are allowed. Clips shorter than three frames create no samples; empty training/validation datasets are rejected.
- Original dimensions must be positive and finite; model input dimensions must be positive multiples of 8. Image readability and GT shape are checked when a sample is read.

Split by independent matches/clips and check actual frame overlap. Different directory paths alone do not establish independence.

## Training

### TrackNet V1

```powershell
python trainV1.py --train_path assets/train --val_path assets/val --exps_path exps --exp_id run_v1_01
```

### TrackNet V2

```powershell
python trainV2.py --train_dataset_path assets/train --val_dataset_path assets/val --num_epochs 10 --val_intervals 2 --exps_path exps/run_v2_01 --logs_path runs/run_v2_01
```

Add `--parallel` to request DataParallel; it is a flag without a `True` value.
Use positive `--num_epochs` and `--val_intervals`. Validation runs every interval and on the final epoch: the example evaluates epochs 2, 4, 6, 8 and 10. The first finite F1 saves `best_model.pth`, including F1 zero; later saves require improvement. `last_model.pth` is saved each epoch. Existing best/last checkpoint paths stop startup: choose a new `--exps_path`.

Optional `--model_path` initializes only the model `state_dict`. It does **not** resume optimizer, scheduler, epoch or best F1; those restart. Use a new output experiment directory even when initializing from previous weights.

V2 epoch losses are weighted by sample count, including the final partial batch. Validation uses one logits forward per batch and derives probabilities with sigmoid. Its metrics aggregate all three temporal channels from overlapping windows; video inference uses only the current channel. A visible prediction outside the distance tolerance counts as FP without an additional FN. Document this protocol when comparing results.

## Video inference

### TrackNet V1

```powershell
python inference_video.py --video_paths input.mp4 --model_path exps/run_v1_01/model_best.pt --output_path output_v1.mp4
```

`--video_paths` accepts one to four input videos.

### TrackNet V2

```powershell
python predictV2.py --video_path input.mp4 --model_path exps/run_v2_01/best_model.pth --video_out output_v2.mp4
```

Input and output must refer to different files. Processing reads until EOF; declared frame count controls only progress. Invalid FPS falls back to 30, invalid video dimensions raise an error, and capture/writer resources are released on failure. Temporal inputs retain clean frames before annotations are drawn.

V2 inference uses the same RGB, PIL bilinear resizing and ImageNet normalization as the dataset. `--input_width`/`--input_height` must be positive multiples of 8, `--threshold` must be in [0, 1], and `--trail_length` can be zero or positive. Hough parameters and evaluation tolerance still require calibration for the data.

## Notebook and visualization

[Notebooks/Trackv2.ipynb](Notebooks/Trackv2.ipynb) imports the current model, dataset, loss and validation. Configure the dataset, original/input dimensions and an explicit checkpoint before predicting or validating. It does not train a model.

V2 helpers in [utils/helpers.py](utils/helpers.py) pair the central RGB frame with its central GT/probability map. They preserve inputs and explicit output mode; denormalization returns a copy. Batch visualization accepts a pair or triple, limits the first batch and returns a grid without displaying it automatically.

## Checks and logs

```powershell
python -B -m unittest discover -s tests -v
python trainV2.py --help
python predictV2.py --help
tensorboard --logdir runs/run_v2_01
```

For V1 TensorBoard logs, use `exps/run_v1_01/plots`. Tests use synthetic fixtures; tracking quality on real matches requires separate evaluation.
