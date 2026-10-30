from .model import BallTrackerNet
import torch
from .dataset import TracknetV1Dataset
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from .train_val import train, validate


def run_training(config):
    """Train V1 from explicit training and validation dataset paths."""
    device = config["device"]
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but is not available")

    train_path, val_path = config["train_path"], config["val_path"]
    for option, path in (("train_path", train_path), ("val_path", val_path)):
        if path is None or not os.path.isdir(path):
            raise ValueError(f"{option} must be an existing directory: {path}")
    if os.path.samefile(train_path, val_path):
        raise ValueError("train_path and val_path must be different directories")
    if config["val_intervals"] <= 0:
        raise ValueError("val_intervals must be greater than zero")
    if config["num_epochs"] <= 0:
        raise ValueError("num_epochs must be greater than zero")

    print("=" * 40)
    print("Starting training")
    print("=" * 40)
    print("Using separate paths:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    train_dataset = TracknetV1Dataset(
        frames_path=os.path.join(train_path, "frames_out"),
        gts_path=os.path.join(train_path, "gts"),
        labels_path=os.path.join(train_path, "labels"),
    )
    val_dataset = TracknetV1Dataset(
        frames_path=os.path.join(val_path, "frames_out"),
        gts_path=os.path.join(val_path, "gts"),
        labels_path=os.path.join(val_path, "labels"),
    )
    for option, dataset, path in (("train_path", train_dataset, train_path), ("val_path", val_dataset, val_path)):
        if len(dataset) == 0:
            raise ValueError(f"{option} contains no temporal samples: {path}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)


    model = BallTrackerNet()
    model = model.to(device)

    if config['model_path'] is not None:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print(f"Model loaded from {config['model_path']}")


    exps_path = os.path.join(config['exps_path'], config['exp_id'])
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    val_best_metric = 0

    print("=" * 40)
    print(f"Starting training on device: {device}")
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Experiment ID: {config['exp_id']}")
    print("=" * 40)

    for epoch in range(config['num_epochs']):
        train_loss = train(model, train_loader, optimizer, device, epoch, config['steps_per_epoch'])
        print('train loss = {}'.format(train_loss))
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) & (epoch % config['val_intervals'] == 0):
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            print('val loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)

    log_writer.close()
