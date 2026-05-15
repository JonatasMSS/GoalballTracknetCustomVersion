
import argparse
from collections import deque
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.TracknetV2 import TrackNet

def preprocess_frame(frame, input_size, device, mean, std):
    """Resize and normalize a BGR frame to match training preprocessing."""
    frame_tensor = torch.from_numpy(frame).to(device)
    frame_tensor = frame_tensor.permute(2, 0, 1).float() / 255.0
    frame_tensor = F.interpolate(
        frame_tensor.unsqueeze(0),
        size=input_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    frame_tensor = (frame_tensor - mean) / std
    return frame_tensor


def build_input(frames, input_size, device, mean, std):
    preprev = preprocess_frame(frames[0], input_size, device, mean, std)
    prev = preprocess_frame(frames[1], input_size, device, mean, std)
    current = preprocess_frame(frames[2], input_size, device, mean, std)
    return torch.cat([preprev, prev, current], dim=0).unsqueeze(0)


def find_ball_center(heatmap, threshold, hough_params):
    heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
    _, thresh = cv2.threshold(heatmap_uint8, threshold, 255, cv2.THRESH_BINARY)
    blurred = cv2.medianBlur(thresh, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=hough_params["dp"],
        minDist=hough_params["min_dist"],
        param1=hough_params["param1"],
        param2=hough_params["param2"],
        minRadius=hough_params["min_radius"],
        maxRadius=hough_params["max_radius"],
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    best_circle = max(circles, key=lambda c: c[2])
    return int(best_circle[0]), int(best_circle[1])


def draw_trajectory(frame, points, color=(0, 0, 255)):
    for point in points:
        cv2.circle(frame, point, radius=3, color=color, thickness=-1)
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], color, thickness=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackNet V2 Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to a pre-trained model")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--video_out", type=str, default="output_video.mp4", help="Path to save output video")
    parser.add_argument("--input_width", type=int, default=640, help="Model input width")
    parser.add_argument("--input_height", type=int, default=360, help="Model input height")
    parser.add_argument("--threshold", type=int, default=200, help="Heatmap threshold 0-255")
    parser.add_argument("--trail_length", type=int, default=4, help="Max trajectory points to draw")
    parser.add_argument("--hough_dp", type=float, default=1.2, help="HoughCircles dp")
    parser.add_argument("--hough_min_dist", type=int, default=10, help="HoughCircles minDist")
    parser.add_argument("--hough_param1", type=int, default=50, help="HoughCircles param1")
    parser.add_argument("--hough_param2", type=int, default=8, help="HoughCircles param2")
    parser.add_argument("--hough_min_radius", type=int, default=1, help="HoughCircles minRadius")
    parser.add_argument("--hough_max_radius", type=int, default=15, help="HoughCircles maxRadius")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrackNet().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    input_size = (args.input_height, args.input_width)
    mean = torch.tensor([0.1307, 0.1307, 0.1307], device=device).view(3, 1, 1)
    std = torch.tensor([0.3081, 0.3081, 0.3081], device=device).view(3, 1, 1)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(args.video_out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (original_width, original_height))

    prev_frames = deque(maxlen=2)
    trajectory = deque(maxlen=args.trail_length)

    hough_params = {
        "dp": args.hough_dp,
        "min_dist": args.hough_min_dist,
        "param1": args.hough_param1,
        "param2": args.hough_param2,
        "min_radius": args.hough_min_radius,
        "max_radius": args.hough_max_radius,
    }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with torch.no_grad():
        for frame_idx in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < 2:
                frames = [frame, frame, frame]
            else:
                frames = [prev_frames[0], prev_frames[1], frame]

            input_tensor = build_input(frames, input_size, device, mean, std)
            heatmap = model(input_tensor).squeeze(0).squeeze(0).detach().cpu().numpy()

            center = find_ball_center(heatmap, args.threshold, hough_params)
            if center is not None:
                scale_x = original_width / input_size[1]
                scale_y = original_height / input_size[0]
                center_scaled = (int(center[0] * scale_x), int(center[1] * scale_y))
                trajectory.append(center_scaled)
                cv2.circle(frame, center_scaled, radius=4, color=(0, 0, 255), thickness=-1)

            draw_trajectory(frame, list(trajectory), color=(0, 0, 255))

            out.write(frame)

            prev_frames.append(frame)

    cap.release()
    out.release()