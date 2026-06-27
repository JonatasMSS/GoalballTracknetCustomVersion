# Tracknet Goalball

## Overview
**Tracknet Goalball** is a unified framework designed for training and evaluating ball-tracking models on goalball datasets. This project brings together **TrackNet V1** and **TrackNet V2** architectures into a single repository, providing a unified and consistent environment for researchers, developers, and sports analysts to experiment with ball-tracking in the context of goalball.

By unifying both versions, this project makes it easier to compare performance, share dataset pipelines, and eventually deploy inference for match analysis.

---

## 🛠️ Environment Setup

It is highly recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Tracknet Goalball"
   ```

2. **Install dependencies:**
   The project requires PyTorch and other standard computer vision libraries. Install them via `pip`:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: Depending on your hardware, you may want to install a specific version of PyTorch with CUDA support. Visit the [PyTorch website](https://pytorch.org/) for instructions.)*

---

## 📂 Dataset Structure

The models expect the goalball dataset to be located in the `assets/dataset` directory. Ensure your dataset is structured as follows before beginning training:

```text
Tracknet Goalball/
└── assets/
    └── dataset/
        ├── frames_out/   # Extracted video frames (.jpg or .png)
        ├── gts/          # Ground truth heatmaps/data
        └── labels/       # CSV or text files containing (x, y) coordinates
```

---

## 🚀 Training

You can train either TrackNet V1 or V2 using the provided training scripts. Both scripts use similar argument structures.

### Training TrackNet V1
```bash
python trainV1.py \
    --batch_size 2 \
    --num_epochs 500 \
    --lr 1e-4 \
    --dataset_path assets/dataset \
    --exps_path exps \
    --exp_id run_v1_01
```

### Training TrackNet V2
TrackNet V2 adds support for DataParallel (multiple GPUs).
```bash
python trainV2.py \
    --batch_size 4 \
    --num_epochs 50 \
    --lr 1e-4 \
    --dataset_path assets/dataset \
    --exps_path exps \
    --logs_path runs/logs \
    --parallel True
```

**Key Arguments:**
- `--batch_size`: Number of samples per batch.
- `--num_epochs`: Total epochs for training.
- `--model_path`: (Optional) Path to a pre-trained `.pt` or `.pth` weight file to resume training.

---

## 🎥 Inference & Prediction

Once you have a trained model, you can run inference on videos to visualize the ball tracking.

### Inference with TrackNet V1
Use `inference_video.py` to run predictions using a TrackNet V1 trained model.
```bash
python inference_video.py --video_path path/to/video.mp4 --model_path exps/run_v1_01/model_best.pt
```
*(Check the script for additional arguments like output video path)*

### Inference with TrackNet V2
Use `predictV2.py` for TrackNet V2 inference.
```bash
python predictV2.py --video_path path/to/video.mp4 --model_path exps/best_model.pth
```

---

## 📈 TensorBoard Logging
Training metrics (Loss, Precision, Recall, F1-Score) are logged automatically. You can view them using TensorBoard:

```bash
# For V1
tensorboard --logdir exps/run_v1_01/plots

# For V2
tensorboard --logdir runs/logs
```

---

## 🤝 Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.
