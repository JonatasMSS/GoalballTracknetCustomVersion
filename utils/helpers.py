
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

def show_grid_batches(dataloader, batch_size=3):
    """Exibe um grid de imagens do batch."""

    first_row = []
    second_row = []
    third_row = []

    for x, y in dataloader:
        first_row = x[:, 6:9, :, :]
        second_row = y
        # Exibe o grid

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Desnormaliza um tensor de imagem."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def make_prediction_grid(model, x):
    """
    x: tensor (3, 9, H, W)

    Retorna:
        grid -> imagem pronta para matplotlib
    """

    model.eval()

    with torch.no_grad():

        # -----------------------------------
        # Predição
        # -----------------------------------
        out = model(x)  # (3,1,H,W)

        # -----------------------------------
        # Pegar frame RGB central
        # (3 canais centrais)
        # -----------------------------------
        rgb_frames = x[:, 3:6]  # (3,3,H,W)

        # Desnormalizar
        rgb_frames = torch.stack([
            denormalize(img)
            for img in rgb_frames
        ])

        # -----------------------------------
        # Heatmaps -> 3 canais
        # -----------------------------------
        out_rgb = out.repeat(1, 3, 1, 1)

        # -----------------------------------
        # Junta:
        #
        # RGB1 RGB2 RGB3
        # OUT1 OUT2 OUT3
        # -----------------------------------
        all_imgs = torch.cat([rgb_frames, out_rgb], dim=0)


        print(f"RGB frames shape: {rgb_frames.shape}, Output shape: {out.shape}, All images shape: {all_imgs.shape}")
        print(f"Max value in RGB frames: {rgb_frames.max().item()}, Min value in RGB frames: {rgb_frames.min().item()}")
        print(f"Max value in Output: {out.max().item()}, Min value in Output: {out.min().item()}")
        # -----------------------------------
        # Grid 2x3
        # -----------------------------------
        grid = make_grid(
            all_imgs,
            nrow=3,
            normalize=False
        )

        return grid

