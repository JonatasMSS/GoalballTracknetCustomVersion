
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets.TracknetV2Dataset import TracknetV2Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid,save_image
import os
from utils import helpers
import cv2  # pip install opencv-python


# ─────────────────────────────────────────────
# 1. BLOCO AUXILIAR
# ─────────────────────────────────────────────

def conv_block(in_ch, out_ch, kernel=3, padding=1):
    """Conv → BN → ReLU (duas vezes) — tijolo básico do encoder."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel, padding=padding),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ─────────────────────────────────────────────
# 2. MODELO TRACKNET
# ─────────────────────────────────────────────

class TrackNet(nn.Module):
    """
    Encoder–Decoder estilo U-Net com backbone VGG-like.

    Parâmetros
    ----------
    in_frames : int
        Número de frames de entrada (padrão 3).
        Canais de entrada = in_frames * 3  (RGB por frame).
    base_ch : int
        Número de filtros no primeiro bloco (dobra a cada nível).
    """

    def __init__(self, in_frames: int = 3, base_ch: int = 64):
        super().__init__()
        self.model_train_info = {
            "train_data":[],
            "val_data":[],
            "epoch":[]
        }
        in_ch = in_frames * 3  # 3 frames × 3 canais = 9

        # ── Encoder ──────────────────────────────
        self.enc1 = conv_block(in_ch,       base_ch)      # 9  → 64
        self.enc2 = conv_block(base_ch,     base_ch * 2)  # 64 → 128
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)  # 128 → 256
        self.enc4 = conv_block(base_ch * 4, base_ch * 8)  # 256 → 512

        self.pool = nn.MaxPool2d(2, 2)

        # ── Bottleneck ───────────────────────────
        self.bottleneck = conv_block(base_ch * 8, base_ch * 16)  # 512 → 1024

        # ── Decoder ──────────────────────────────
        self.dec4_conv = conv_block(base_ch * 16 + base_ch * 8, base_ch * 8)
        self.dec3_conv = conv_block(base_ch * 8  + base_ch * 4, base_ch * 4)
        self.dec2_conv = conv_block(base_ch * 4  + base_ch * 2, base_ch * 2)
        self.dec1_conv = conv_block(base_ch * 2  + base_ch,     base_ch)

        # ── Cabeça de saída ──────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  
        )
    
        # self._init_output_layer()

    # def _init_output_layer(self):
    #     """
    #     Inicializa a cabeça de saída para valores próximos a 0.5.
    #     Isso faz com que pré-treino gere gaussianas fracas, não ruído puro.
    #     """
    #     # Inicializa último layer com valores pequenos e negativos
    #     # Assim, sigmoid(x) ≈ 0.4-0.5 (não ruído branco)
    #     with torch.no_grad():
    #         final_layer = self.head[-1]  # último Conv2d
    #         final_layer.weight.fill_(0.01)
    #         if final_layer.bias is not None:
    #             final_layer.bias.fill_(-2.0)  # sigmoid(-2) ≈ 0.12


    def _up_and_concat(self, x, skip):
        """Faz upsample de x para o tamanho de skip, depois concatena."""
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)

    def save_epoch_info(self,train_loss, val_loss, epoch):
        self.model_train_info['train_data'].append(train_loss)
        self.model_train_info['val_data'].append(val_loss)
        self.model_train_info['epoch'].append(epoch)
    
    def forward(self, x):
        # ── Encoder ──
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # ── Bottleneck ──
        b  = self.bottleneck(self.pool(e4))

        # ── Decoder ──
        d4 = self.dec4_conv(self._up_and_concat(b,  e4))
        d3 = self.dec3_conv(self._up_and_concat(d4, e3))
        d2 = self.dec2_conv(self._up_and_concat(d3, e2))
        d1 = self.dec1_conv(self._up_and_concat(d2, e1))

        return torch.sigmoid(self.head(d1))  # (B, 1, H, W) ∈ [0, 1]


if __name__ == "__main__":
    model = TrackNet().to("cpu")
    DIR = "assets/dataset/"
    frames = os.path.join(DIR, "frames_out")
    gts = os.path.join(DIR,"gts")

    labels = os.path.join(DIR,"labels")

    dataset = TracknetV2Dataset(frames_path=frames, gts_path=gts, labels_path=labels, debug=False)
    x, y, x_pos, y_pos, vis = dataset[0]

    input = x.unsqueeze(0)  # (1, 9, H, W)
    output = model(input)   # (1, 1, H, W)
    print(f"Input shape: {input.shape}, Output shape: {output.shape}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    input =  helpers.denormalize(input.squeeze(0)[6:9])  # (3, H, W)

    axs[0].imshow(input.permute(1, 2, 0).cpu().numpy())
    axs[0].set_title("Input (Frame 3)")
    axs[0].axis('off')

    axs[1].imshow(output.squeeze(0).squeeze(0).detach().cpu().numpy(), cmap='gray')
    axs[1].set_title("Output (Predicted Heatmap)")
    axs[1].axis('off')
    plt.show()

    

