
import argparse
from collections import deque
import math
import os

import cv2
import torch
from torchvision.transforms import InterpolationMode, functional as TF
from tqdm import tqdm

from models.TracknetV2 import TrackentV2
from utils.TracknetV2.train_val import postprocessing

def preprocess_frame(frame, input_size, device, mean, std):
    """Resize and normalize a BGR frame to match training preprocessing."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_image = TF.to_pil_image(frame_rgb)
    frame_image = TF.resize(frame_image, input_size, interpolation=InterpolationMode.BILINEAR)
    frame_tensor = TF.to_tensor(frame_image).to(device)
    frame_tensor = (frame_tensor - mean) / std
    return frame_tensor


def build_input(frames, input_size, device, mean, std):
    preprev = preprocess_frame(frames[0], input_size, device, mean, std)
    prev = preprocess_frame(frames[1], input_size, device, mean, std)
    current = preprocess_frame(frames[2], input_size, device, mean, std)
    return torch.cat([preprev, prev, current], dim=0).unsqueeze(0)


def draw_trajectory(frame, points, color=(0, 0, 255)):
    for point in points:
        cv2.circle(frame, point, radius=3, color=color, thickness=-1)


def main():
    parser = argparse.ArgumentParser(description="TrackNet V2 Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a pre-trained model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--video_out", type=str, default="output_video.mp4", help="Path to save output video")
    parser.add_argument("--input_width", type=int, default=640, help="Model input width")
    parser.add_argument("--input_height", type=int, default=360, help="Model input height")
    parser.add_argument("--threshold", type=float, default=0.5, help="Ball probability threshold (0-1)")
    parser.add_argument("--trail_length", type=int, default=4, help="Max trajectory points to draw")

    args = parser.parse_args()

    if args.input_width <= 0 or args.input_height <= 0 or args.input_width % 8 or args.input_height % 8:
        parser.error('--input_width and --input_height must be positive multiples of 8')
    if not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.trail_length < 0:
        parser.error('--trail_length must be greater than or equal to zero')

    input_path = os.path.normcase(os.path.abspath(args.video_path))
    output_path = os.path.normcase(os.path.abspath(args.video_out))
    same_file = input_path == output_path
    if not same_file and os.path.exists(args.video_path) and os.path.exists(args.video_out):
        same_file = os.path.samefile(args.video_path, args.video_out)
    if same_file:
        parser.error("--video_path and --video_out must refer to different files")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackentV2(mode="inference").to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    input_size = (args.input_height, args.input_width)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    cap = cv2.VideoCapture(args.video_path)
    out = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {args.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not math.isfinite(width) or not math.isfinite(height) or width < 1 or height < 1:
            raise ValueError(f"Invalid video dimensions: {width}x{height}")
        original_width, original_height = int(width), int(height)

        os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(args.video_out, fourcc, fps, (original_width, original_height))
        if not out.isOpened():
            raise ValueError(f"Unable to open output video: {args.video_out}")

        prev_frames = deque(maxlen=2)
        trajectory = deque(maxlen=args.trail_length)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else None

        with torch.no_grad(), tqdm(total=total_frames) as progress:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frames = list(prev_frames) + [frame]
                while len(frames) < 3:
                    frames.insert(0, frames[0])

                input_tensor = build_input(frames, input_size, device, mean, std)
                prev_frames.append(frame.copy())
                current_heatmap = model(input_tensor)[0, 2].unsqueeze(0)

                center = postprocessing(current_heatmap, threshold=args.threshold)
                if center is not None:
                    scale_x = original_width / input_size[1]
                    scale_y = original_height / input_size[0]
                    center_scaled = (int(center[0] * scale_x), int(center[1] * scale_y))
                    trajectory.append(center_scaled)
                    cv2.circle(frame, center_scaled, radius=4, color=(0, 0, 255), thickness=-1)

                draw_trajectory(frame, list(trajectory), color=(0, 0, 255))
                out.write(frame)
                progress.update(1)
    finally:
        cap.release()
        if out is not None:
            out.release()


if __name__ == "__main__":
    main()
