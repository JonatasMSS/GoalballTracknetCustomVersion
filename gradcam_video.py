import argparse
from collections import deque
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from inference_video import get_input_tensor
from models.TracknetV1 import BallTrackerNet


def make_gradcam(output, activation, model):
    predicted_classes = output.argmax(dim=1)[0]
    pixel_index = predicted_classes.argmax()
    class_index = predicted_classes[pixel_index]

    if class_index.item() == 0:
        return np.zeros(activation.shape[-2:], dtype=np.float32)

    model.zero_grad(set_to_none=True)
    output[0, class_index, pixel_index].backward()
    if activation.grad is None:
        raise RuntimeError("Grad-CAM did not receive gradients from the target layer.")

    weights = activation.grad[0].mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * activation[0]).sum(dim=0))
    cam -= cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam /= maximum
    return cam.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Create a TrackNetV1 Grad-CAM overlay video.")
    parser.add_argument("--video_path", required=True, help="Input video path.")
    parser.add_argument("--model_path", required=True, help="TrackNetV1 .pt weights path.")
    parser.add_argument("--output_path", default="gradcam_output.mp4", help="Output video path.")
    parser.add_argument("--opacity", type=float, default=0.25, help="Heatmap opacity from 0 to 1.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not 0 <= args.opacity <= 1:
        parser.error("--opacity must be between 0 and 1")
    if not os.path.exists(args.video_path):
        parser.error(f"video not found: {args.video_path}")
    if not os.path.exists(args.model_path):
        parser.error(f"model not found: {args.model_path}")

    model = BallTrackerNet().to(args.device)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.eval()

    captured = {}

    def capture_activation(_module, _inputs, output):
        output.retain_grad()
        captured["activation"] = output

    hook = model.conv17.register_forward_hook(capture_activation)
    capture = cv2.VideoCapture(args.video_path)
    writer = None

    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {args.video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0:
            raise RuntimeError("Input video has invalid dimensions or FPS.")

        writer = cv2.VideoWriter(
            args.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {args.output_path}")

        frames = deque(maxlen=3)
        progress = tqdm(total=total or None, desc="Grad-CAM")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            resized = cv2.resize(frame, (640, 360))
            frames.append(resized)
            while len(frames) < 3:
                frames.appendleft(resized)

            input_tensor = get_input_tensor(frames[-1], frames[-2], frames[-3]).to(args.device)
            model.zero_grad(set_to_none=True)
            output = model(input_tensor)
            cam = make_gradcam(output, captured["activation"], model)

            cam = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame, 1 - args.opacity, heatmap, args.opacity, 0)
            writer.write(overlay)
            progress.update(1)

        progress.close()
    finally:
        hook.remove()
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Grad-CAM video saved to '{args.output_path}'.")


if __name__ == "__main__":
    main()
