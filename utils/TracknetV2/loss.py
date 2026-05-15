import torch


def wbce_loss(predictions, targets, pos_weight=10.0, neg_weight=0.5):
    predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)
    
    loss = -pos_weight * targets * torch.log(predictions) - \
           neg_weight * (1 - targets) * torch.log(1 - predictions)
    
    return loss.mean()

