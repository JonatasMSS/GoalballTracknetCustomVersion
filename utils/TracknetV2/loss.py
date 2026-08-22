import torch
import torch.nn.functional as F






def wbce_loss(logits, targets, gamma=2.0):
    """
    WBCE dinâmica no estilo TrackNetV2.

    logits:
        saída bruta do modelo, SEM sigmoid
        shape: [B, C, H, W]

    targets:
        heatmaps ground truth em [0, 1]
        shape: [B, C, H, W]

    gamma:
        controla o peso dado aos exemplos difíceis.
        TrackNetV2 usa comportamento equivalente a gamma=2.
    """

    # Probabilidade prevista
    probs = torch.sigmoid(logits)

    # BCE estável numericamente
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    # Probabilidade da classe correta
    pt = (
        probs * targets +
        (1 - probs) * (1 - targets)
    )

    # Peso depende da própria predição
    weight = (1 - pt) ** gamma

    loss = weight * bce

    return loss.mean()