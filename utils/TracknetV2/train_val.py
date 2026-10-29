

import numpy as np
import cv2
import torch


def _set_model_mode(model, mode):
    """Set TrackNet's output mode, including when wrapped by DataParallel."""
    target_model = model.module if hasattr(model, "module") else model
    target_model.set_mode(mode)

def train(model, train_loader, criterion, optimizer, device, batch_shown = 1):
    model.train()
    _set_model_mode(model, "train")
    running_loss = 0.0
    sample_count = 0

    for batch_idx,(x,y,_) in enumerate(train_loader):

        
        inputs, targets = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        

        if batch_idx % batch_shown == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        

        running_loss += loss.item() * inputs.size(0)
        sample_count += inputs.size(0)
    if sample_count == 0:
        raise ValueError("Training loader contains no samples")
    return running_loss / sample_count


def get_circle_center_hough(
    binary_image,
    min_radius=1,
    max_radius=50,
    min_dist=20,
    param1=50,
    param2=10
):


    # Garante uint8
    image = binary_image.astype(np.uint8)

    # Suavização para ajudar o Hough
    blurred = cv2.GaussianBlur(
        image,
        (9, 9),
        2
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Primeiro círculo detectado
    x, y, radius = circles[0]

    return x, y


def postprocessing(prediction, threshold=0.5):
    # Apply sigmoid to get probabilities
    prob_map = prediction

    # Convert to numpy array and squeeze to [H, W]
    prob_map_np = prob_map.squeeze(0).cpu().numpy()

    # Thresholding
    binary_map = (prob_map_np > threshold).astype(np.uint8) * 255

    # Use Hough Circle Transform to detect circles
    center = get_circle_center_hough(binary_map)

    return center


def validate(
    model,
    val_loader,
    criterion,
    device,
    writer=None,
    epoch=None,
    max_distance=10,
    threshold=0.5,
):
    model.eval()
    val_loss = 0.0
    sample_count = 0
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}


    with torch.no_grad():
        for x, y, labels in val_loader:
            inputs, targets = x.to(device), y.to(device)

            _set_model_mode(model, "train")
            logits = model(inputs)
            loss = criterion(logits, targets)

            print(f"Validation Batch Loss: {loss.item():.4f}")
            val_loss += loss.item() * inputs.size(0)
            sample_count += inputs.size(0)

            _set_model_mode(model, "inference")
            probabilities = torch.sigmoid(logits)

            for batch_idx in range(probabilities.size(0)):
                for channel_idx in range(probabilities.size(1)):
                    prediction = probabilities[batch_idx, channel_idx].unsqueeze(0)
                    center = postprocessing(prediction, threshold=threshold)
                    gt_x, gt_y, visibility = (
                        labels[channel_idx][0][batch_idx].item(),
                        labels[channel_idx][1][batch_idx].item(),
                        labels[channel_idx][2][batch_idx].item(),
                    )

                    if center is None:
                        counts["fn" if visibility == 1 else "tn"] += 1
                        continue

                    if visibility != 1:
                        counts["fp"] += 1
                        continue

                    distance = np.hypot(center[0] - gt_x, center[1] - gt_y)
                    counts["tp" if distance <= max_distance else "fp"] += 1

    if sample_count == 0:
        raise ValueError("Validation loader contains no samples")

    eps = 1e-6
    precision = counts["tp"] / (counts["tp"] + counts["fp"] + eps)
    recall = counts["tp"] / (counts["tp"] + counts["fn"] + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    total = sum(counts.values())
    accuracy = (counts["tp"] + counts["tn"]) / (total + eps)
    avg_val_loss = val_loss / sample_count

    print(
        "Validation metrics | "
        f"TP: {counts['tp']}, TN: {counts['tn']}, "
        f"FP: {counts['fp']}, FN: {counts['fn']}, total: {total} | "
        f"accuracy: {accuracy:.4f}, precision: {precision:.4f}, "
        f"recall: {recall:.4f}, f1: {f1_score:.4f}"
    )

    if writer is not None and epoch is not None:
        for metric, value in counts.items():
            writer.add_scalar(f"Metrics/{metric.upper()}", value, epoch)
        writer.add_scalar("Metrics/Total", total, epoch)
        writer.add_scalar("Metrics/Accuracy", accuracy, epoch)


    return avg_val_loss, precision, recall, f1_score
            




