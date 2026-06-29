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
    parser.add_argument('--video_paths', nargs='+', type=str, required=True, help='Path to one or more input videos (max 4).')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pt file).')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to save the output video.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on (cuda/cpu).')
    parser.add_argument('--trail_length', type=int, default=7, help='Length of the ball trajectory trail.')
    parser.add_argument('--start_time', type=float, default=0.0, help='Time in seconds to start processing from.')
    parser.add_argument('--max_seconds', type=float, default=None, help='Maximum number of seconds to process.')
    args = parser.parse_args()

    num_videos = len(args.video_paths)
    if num_videos < 1 or num_videos > 4:
        print("Error: Please provide between 1 and 4 video paths.")
        return

    for vp in args.video_paths:
        if not os.path.exists(vp):
            print(f"Error: Input video '{vp}' not found.")
            return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        return

    print(f"Loading model from {args.model_path} onto {args.device}...")
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    caps = [cv2.VideoCapture(vp) for vp in args.video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open video '{args.video_paths[i]}'.")
            return

    # Use the first video for original dimensions and output properties
    original_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    
    if args.start_time > 0:
        start_frame_idx = int(args.start_time * fps)
        for cap in caps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)

    # Total frames is the maximum among all videos
    total_frames = max([int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps])
    if args.start_time > 0:
        total_frames = max(0, total_frames - start_frame_idx)

    if args.max_seconds is not None:
        max_frames_from_seconds = int(args.max_seconds * fps)
        total_frames = min(total_frames, max_frames_from_seconds)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(args.output_path, fourcc, fps, (original_width, original_height))

    # Buffers and trails for each video
    frame_buffers = [[] for _ in range(num_videos)]
    trails = [deque(maxlen=args.trail_length) for _ in range(num_videos)]
    
    # Store black frames for when a video ends
    black_frame = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    print(f"Processing {num_videos} video(s).")
    print(f"Output Resolution: {original_width}x{original_height}, Max Frames: {total_frames}")

    with torch.no_grad():
        for _ in tqdm(range(total_frames), desc="Inference"):
            frames = []
            valid_idx = []
            
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    frame = black_frame.copy()
                frames.append(frame)
                valid_idx.append(ret)

            original_frames = [f.copy() for f in frames]

            input_tensors = []
            for i, frame in enumerate(frames):
                resized_frame = cv2.resize(frame, (640, 360))

                if len(frame_buffers[i]) == 0:
                    frame_buffers[i] = [resized_frame, resized_frame, resized_frame]
                else:
                    frame_buffers[i].pop(0)
                    frame_buffers[i].append(resized_frame)

                # Note: TracknetDataset orders frames as: [current, prev, pre_prev]
                # Since frame_buffer is [pre_prev, prev, current]
                # We pass them as: current (index 2), prev (index 1), pre_prev (index 0)
                t = get_input_tensor(frame_buffers[i][2], frame_buffers[i][1], frame_buffers[i][0])
                input_tensors.append(t)

            # Process each input sequentially to avoid GPU Out-Of-Memory errors when using 3 or 4 videos
            feature_maps = []
            for t in input_tensors:
                t = t.to(args.device)
                out = model(t)
                fm = out.argmax(dim=1).detach().cpu().numpy()[0]
                feature_maps.append(fm)

            for i in range(num_videos):
                fm = feature_maps[i]
                x, y = postprocess(fm, original_width, original_height)
                
                # Only add to trail if video hasn't ended and point is detected
                if valid_idx[i] and x is not None and y is not None:
                    trails[i].append((x, y))

                # Draw trail on original_frames[i]
                for j, point in enumerate(trails[i]):
                    intensity = int(255 * (j + 1) / len(trails[i]))
                    color = (0, intensity, 0)
                    radius = max(2, int(6 * (j + 1) / len(trails[i])))
                    
                    cv2.circle(original_frames[i], point, radius=radius, color=color, thickness=-1)
                    
                    if j == len(trails[i]) - 1:
                        cv2.circle(original_frames[i], point, radius=6, color=(0, 255, 0), thickness=-1)
                        cv2.circle(original_frames[i], point, radius=8, color=(0, 0, 0), thickness=2)

            # Combine frames into a single image
            if num_videos == 1:
                combined_frame = original_frames[0]
            else:
                # 2x2 grid
                grid_frame = np.zeros((original_height, original_width, 3), dtype=np.uint8)
                h_half = original_height // 2
                w_half = original_width // 2
                
                for i in range(4):
                    if i < num_videos:
                        resized_quad = cv2.resize(original_frames[i], (w_half, h_half))
                    else:
                        resized_quad = cv2.resize(black_frame, (w_half, h_half))
                        
                    row = i // 2
                    col = i % 2
                    
                    grid_frame[row*h_half : (row+1)*h_half, col*w_half : (col+1)*w_half] = resized_quad
                    
                combined_frame = grid_frame

            out_video.write(combined_frame)

    for cap in caps:
        cap.release()
    out_video.release()
    print(f"\nInference completed. Saved output video to '{args.output_path}'.")

if __name__ == '__main__':
    main()
