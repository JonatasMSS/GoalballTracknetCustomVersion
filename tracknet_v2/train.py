import math
import os

import torch
from .model import TrackentV2
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .loss import wbce_loss
from .dataset import TracknetV2Dataset
from .train_val import train, validate

def run_training(config):
    """Train V2 from a complete, validated configuration dictionary."""
    device = config["device"]
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but is not available")

    train_dataset_path = config["train_dataset_path"]
    val_dataset_path = config["val_dataset_path"]
    for option, path in (
        ('train_dataset_path', train_dataset_path),
        ('val_dataset_path', val_dataset_path),
    ):
        if not os.path.isdir(path):
            raise ValueError(f'{option} must be an existing directory: {path}')
    if os.path.samefile(train_dataset_path, val_dataset_path):
        raise ValueError('train_dataset_path and val_dataset_path must be different directories')
    if config["val_intervals"] <= 0:
        raise ValueError('val_intervals must be greater than zero')
    if config["num_epochs"] <= 0:
        raise ValueError('num_epochs must be greater than zero')

    #Paths

    logs_dir = config["logs_path"]
    weights_dir = config["exps_path"]
    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    last_model_path = os.path.join(weights_dir, 'last_model.pth')
    for checkpoint_path in (best_model_path, last_model_path):
        if os.path.lexists(checkpoint_path):
            raise ValueError(f'Checkpoint already exists: {checkpoint_path}. Use a new exps_path.')

    train_dataset = TracknetV2Dataset(
        frames_path=f"{train_dataset_path}/frames_out",
        gts_path=f"{train_dataset_path}/gts",
        labels_path=f"{train_dataset_path}/labels",
        original_width=config["original_width"],
        original_height=config["original_height"],
    )
    val_dataset = TracknetV2Dataset(
        frames_path=f"{val_dataset_path}/frames_out",
        gts_path=f"{val_dataset_path}/gts",
        labels_path=f"{val_dataset_path}/labels",
        original_width=config["original_width"],
        original_height=config["original_height"],
    )
    for option, dataset, path in (
        ('train_dataset_path', train_dataset, train_dataset_path),
        ('val_dataset_path', val_dataset, val_dataset_path),
    ):
        if len(dataset) == 0:
            raise ValueError(f'{option} contains no temporal samples: {path}')

    if not os.path.exists(logs_dir):
        print(f"Criando diretório para logs: {logs_dir}")
        os.makedirs(logs_dir)
    if not os.path.exists(weights_dir):
        print(f"Criando diretório para pesos: {weights_dir}")
        os.makedirs(weights_dir)



    model = TrackentV2().to(device)




    if config["model_path"] is not None:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
        print(f"Model loaded from {config['model_path']}")

    if config["parallel"] and device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)


    #Model Config
    writer = SummaryWriter(logs_dir)


    loss_function = wbce_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-5)

    stepper = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=config["stepper_gamma"])


    num_epochs = config["num_epochs"]
    epochs_to_validate = config["val_intervals"]

    best_f1_score = None

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function,optimizer, device,batch_shown=1)

        writer.add_scalar('Loss/Train', train_loss, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        if (epoch + 1) % epochs_to_validate == 0 or epoch + 1 == num_epochs:
            val_loss,precision, recall, f1  = validate(model, val_loader, loss_function,device, writer=writer, epoch=epoch)
            if not math.isfinite(f1):
                raise ValueError(f'Validation F1 must be finite at epoch {epoch + 1}: {f1}')
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Metrics/Precision', precision, epoch)
            writer.add_scalar('Metrics/Recall', recall, epoch)
            writer.add_scalar('Metrics/F1_Score', f1, epoch)
            print(f"Epoch {epoch+1}/{num_epochs} \n- Train Loss: {train_loss:.4f} \n- Val Loss: {val_loss:.4f} \n- Precision: {precision:.4f} \n- Recall: {recall:.4f} \n- F1 Score: {f1:.4f}")

            if best_f1_score is None or f1 > best_f1_score:
                best_f1_score = f1
                if config["parallel"] and device == "cuda" and torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with F1 Score: {best_f1_score:.4f}")

        if config["parallel"] and device == "cuda" and torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_model_path)
        else:
            torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved at epoch {epoch+1}")

        stepper.step()  # Update learning rate

    writer.close()
