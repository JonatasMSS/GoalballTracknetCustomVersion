
import torch
from torchvision.utils import make_grid

def show_grid_batches(dataloader, batch_size=3):
    """Grid V2: x [B,9,H,W], y [B,3,H,W]; RGB/GT centrais em duas linhas."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    try:
        x, y = next(iter(dataloader))[:2]
    except StopIteration:
        raise ValueError("Dataloader contains no samples") from None
    count = min(batch_size, x.size(0))
    if count == 0:
        raise ValueError("Batch contains no samples")
    rgb_frames = torch.stack([denormalize(image) for image in x[:count, 3:6]])
    gt_rgb = y[:count, 1:2].repeat(1, 3, 1, 1)
    return make_grid(torch.cat([rgb_frames, gt_rgb]), nrow=count, normalize=False)

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Desnormaliza uma cópia de um tensor RGB [3,H,W]."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def make_prediction_grid(model, x):
    """Grid V2 para x [B,9,H,W] e saída [B,3,H,W], sem alterar x ou mode."""
    if x.size(0) == 0:
        raise ValueError("Batch contains no samples")
    model.eval()
    with torch.no_grad():
        out = model(x)
        target_model = model.module if hasattr(model, "module") else model
        if target_model.mode == "train":
            out = torch.sigmoid(out)
        rgb_frames = torch.stack([denormalize(image) for image in x[:, 3:6]])
        out_rgb = out[:, 1:2].repeat(1, 3, 1, 1)
        return make_grid(torch.cat([rgb_frames, out_rgb]), nrow=x.size(0), normalize=False)
