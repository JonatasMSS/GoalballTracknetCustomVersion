from models.TracknetV1 import BallTrackerNet
import torch
from datasets.TracknetV1Dataset import TracknetV1Dataset
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from utils.TracknetV1.train_val import train, validate
import argparse

if __name__ == '__main__':



    print("=" * 40)
    print("Starting training")
    print("=" * 40)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=2, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=200, help='number of steps per one epoch')
    parser.add_argument('--exps_path', type=str, default='exps', help='path to save results')
    parser.add_argument('--dataset_path', type=str, default='assets/dataset', help='path to dataset')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    parser.add_argument("--train_path", type=str, default=None, help="path to training frames")
    parser.add_argument("--val_path", type=str, default=None, help="path to validation frames")
    args = parser.parse_args()
    

    if args.train_path is not None and args.val_path is not None:
        print(f"Using separate paths:")
        print(f"  Train: {args.train_path}")
        print(f"  Val: {args.val_path}")
        train_dataset = TracknetV1Dataset(
            frames_path=os.path.join(args.train_path, "frames_out"), 
            gts_path=os.path.join(args.train_path, "gt"), 
            labels_path=os.path.join(args.train_path, "labels")
        )
        val_dataset = TracknetV1Dataset(
            frames_path=os.path.join(args.val_path, "frames_out"), 
            gts_path=os.path.join(args.val_path, "gt"), 
            labels_path=os.path.join(args.val_path, "labels")
        )
    else:
        train_dataset = TracknetV1Dataset(frames_path=os.path.join(args.dataset_path, "frames_out"), gts_path=os.path.join(args.dataset_path, "gt"), labels_path=os.path.join(args.dataset_path, "labels"))
        val_dataset = TracknetV1Dataset(frames_path=os.path.join(args.dataset_path, "frames_out"), gts_path=os.path.join(args.dataset_path, "gt"), labels_path=os.path.join(args.dataset_path, "labels"))
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    model = BallTrackerNet()
    device = 'cuda'
    model = model.to(device)

    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")

    
    exps_path = os.path.join(args.exps_path, args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    val_best_metric = 0

    print("=" * 40)
    print(f"Starting training on device: {device}")
    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Validation dataset size: {len(val_dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Total epochs: {args.num_epochs}")
    print(f"Experiment ID: {args.exp_id}")
    print("=" * 40)

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
        print('train loss = {}'.format(train_loss))
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
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
