import argparse
import os

import torch
from models.TracknetV2 import TrackNet
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from utils.TracknetV2.loss import wbce_loss
from datasets.TracknetV2Dataset import TracknetV2Dataset
from utils.TracknetV2.train_val import train, validate

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=1, help='number of epochs to run validation')
    parser.add_argument('--exps_path', type=str, default='exps', help='path to save results')
    parser.add_argument('--logs_path', type=str, default='runs/logs', help='path to save logs')
    parser.add_argument('--dataset_path', type=str, default='assets/dataset', help='path to dataset')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    parser.add_argument('--parallel', type=bool, default=False, help='use multiple GPUs')
    args = parser.parse_args()

    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #Paths
    
    logs_dir = args.logs_path
    weights_dir = args.exps_path

    dataset_dir = args.dataset_path


    if not os.path.exists(logs_dir):
        print(f"Criando diretório para logs: {logs_dir}")
        os.makedirs(logs_dir)
    if not os.path.exists(weights_dir):
        print(f"Criando diretório para pesos: {weights_dir}")
        os.makedirs(weights_dir)
    


    best_model_path = os.path.join(weights_dir, 'best_model.pth')
    last_model_path = os.path.join(weights_dir, 'last_model.pth')
    model = TrackNet().to(device)

    


    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    
    if args.parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    

    #Model Config
    writer = SummaryWriter(logs_dir)


    loss_function = wbce_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)




    
    

    

    dataset = TracknetV2Dataset(frames_path=f"{dataset_dir}/frames_out", gts_path=f"{dataset_dir}/gts", labels_path=f"{dataset_dir}/labels", debug=False)

 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    num_epochs = args.num_epochs
    epochs_to_validate = args.val_intervals

    best_f1_score = 0.0

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, loss_function,optimizer, device)

        writer.add_scalar('Loss/Train', train_loss, epoch)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")
        if epoch > 0 and epoch % epochs_to_validate == 0:
            val_loss,precision, recall, f1  = validate(model, val_loader, loss_function,device, writer=writer, epoch=epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Metrics/Precision', precision, epoch)
            writer.add_scalar('Metrics/Recall', recall, epoch)
            writer.add_scalar('Metrics/F1_Score', f1, epoch)
            print(f"Epoch {epoch+1}/{num_epochs} \n- Train Loss: {train_loss:.4f} \n- Val Loss: {val_loss:.4f} \n- Precision: {precision:.4f} \n- Recall: {recall:.4f} \n- F1 Score: {f1:.4f}")

            if f1 > best_f1_score:
                best_f1_score = f1
                if args.parallel and torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), best_model_path)
                else:
                    torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with F1 Score: {best_f1_score:.4f}")

        if args.parallel and torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), last_model_path)
        else:
            torch.save(model.state_dict(), last_model_path)
        print(f"Last model saved at epoch {epoch+1}")
        
    writer.close()







    
