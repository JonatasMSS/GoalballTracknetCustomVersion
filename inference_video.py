import argparse
import os
import cv2
import torch
import numpy as np
from collections import deque
from tqdm import tqdm

from models.TracknetV1 import BallTrackerNet

def get_input_tensor(img, img_prev, img_preprev):
    """
    Prepares the input tensor for the model as done in TracknetV1Dataset.
    """
    # imgs are already resized to 640x360
    imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
    imgs = imgs.astype(np.float32) / 255.0
    imgs = np.rollaxis(imgs, 2, 0)
    # Add batch dimension
    return torch.from_numpy(imgs).unsqueeze(0)

def postprocess(feature_map, original_width, original_height):
    """
    Processes the model output to find the ball coordinates, scaled to original resolution.
    """
    # feature_map comes from out.argmax(dim=1).detach().cpu().numpy()[0]
    feature_map = feature_map * 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)
    
    x, y = None, None
    if circles is not None:
        if len(circles) == 1:
            x_640 = circles[0][0][0]
            y_360 = circles[0][0][1]
            x = int(x_640 * (original_width / 640.0))
            y = int(y_360 * (original_height / 360.0))
    return x, y

def main():
    parser = argparse.ArgumentParser(description="Run TrackNetV1 inference on a video.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pt file).')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to save the output video.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on (cuda/cpu).')
    parser.add_argument('--trail_length', type=int, default=7, help='Length of the ball trajectory trail.')
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Input video '{args.video_path}' not found.")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    print(f"Loading model from {args.model_path} onto {args.device}...")
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{args.video_path}'.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_path, fourcc, fps, (original_width, original_height))

    # To store frames at 640x360 for model input
    frame_buffer = []
    # To store recent ball coordinates for the trail
    trail = deque(maxlen=args.trail_length)

    print(f"Processing video: {args.video_path}")
    print(f"Original Resolution: {original_width}x{original_height}, Total Frames: {total_frames}")

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Inference"):
            ret, frame = cap.read()
            if not ret:
                break

            # Store the original frame for writing the output
            original_frame = frame.copy()

            # Resize to 640x360 for the model
            resized_frame = cv2.resize(frame, (640, 360))

            # Handle the first two frames by duplicating the first frame
            if len(frame_buffer) == 0:
                frame_buffer = [resized_frame, resized_frame, resized_frame]
            else:
                frame_buffer.pop(0)
                frame_buffer.append(resized_frame)

            # Note: TracknetDataset orders frames as: [current, prev, pre_prev]
            # Since frame_buffer is [pre_prev, prev, current]
            # We pass them as: current (index 2), prev (index 1), pre_prev (index 0)
            input_tensor = get_input_tensor(frame_buffer[2], frame_buffer[1], frame_buffer[0])
            input_tensor = input_tensor.to(args.device)

            # Forward pass
            out = model(input_tensor)
            
            # Postprocess to get coordinates
            feature_map = out.argmax(dim=1).detach().cpu().numpy()[0]
            x, y = postprocess(feature_map, original_width, original_height)

            if x is not None and y is not None:
                trail.append((x, y))
            else:
                # If no ball detected, we don't append to trail.
                pass

            # Draw the trail and current dot
            for i, point in enumerate(trail):
                # Calculate alpha for fading effect (older points are smaller/more transparent)
                # Since we can't easily draw transparent circles in OpenCV without overlays,
                # we will simulate it by changing the color intensity and radius.
                intensity = int(255 * (i + 1) / len(trail))
                color = (0, intensity, 0) # Green fading out
                radius = max(2, int(6 * (i + 1) / len(trail)))
                
                # Draw trail dot
                cv2.circle(original_frame, point, radius=radius, color=color, thickness=-1)
                
                # If it's the most recent point, draw a larger dot or a distinct boundary
                if i == len(trail) - 1:
                    cv2.circle(original_frame, point, radius=6, color=(0, 255, 0), thickness=-1)
                    cv2.circle(original_frame, point, radius=8, color=(0, 0, 0), thickness=2)

            out_video.write(original_frame)

    cap.release()
    out_video.release()
    print(f"\nInference completed. Saved output video to '{args.output_path}'.")

if __name__ == '__main__':
    main()
