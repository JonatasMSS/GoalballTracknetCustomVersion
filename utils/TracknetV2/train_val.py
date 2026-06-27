

import torch
import cv2
import numpy as np
from models.TracknetV2 import TrackNet
from datasets.TracknetV2Dataset import TracknetV2Dataset
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

def train(model, train_loader, criterion, optimizer, device, batch_shown = 10):
    model.train()
    running_loss = []

    for batch_idx, (inputs, targets,_,_,_ ) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % batch_shown == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        

        running_loss.append(loss.item())
    avg_loss = sum(running_loss) / len(running_loss)
    return avg_loss

def postprocess(feature_map):
    feature_map = cv2.threshold(feature_map, 0.7, 1.0, cv2.THRESH_BINARY)[1]  # Binarize the output
    x, y = None, None
    feature_map = (feature_map * 255).astype(np.uint8)  # Convert to uint8 for HoughCircles
    circles = cv2.HoughCircles(feature_map, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=60, param2=10, minRadius=2, maxRadius=7)
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]
            y = circles[0][0][1]
    return x, y   


def postprocess_grid(gt_img,feat_map, threshold=0.7):
    pure_img = feat_map.copy()
    feature_map = cv2.threshold(feat_map, threshold, 1.0, cv2.THRESH_BINARY)[1]  # Binarize the output
    feature_map = (feature_map * 255).astype(np.uint8)  # Convert to uint8 for HoughCircles
    grid_size = 2

    x,y = postprocess(pure_img)

    fig, axs = plt.subplots(2, grid_size, figsize=(10, 5))
    axs[0,0].imshow(gt_img, cmap='gray')
    axs[0,0].set_title('Ground Truth')


    axs[0,1].imshow(feature_map, cmap='gray')
    axs[0,1].set_title('Feature Map (Binarized)')

    axs[1,0].imshow(pure_img, cmap='gray')
    axs[1,0].set_title('Pure Feature Map')

    axs[1,1].imshow(feature_map, cmap='gray')
    # axs[1,1].imshow(gt_img, cmap='jet', alpha=0.5)
    axs[1,1].scatter(x, y, color='green', s=100, edgecolors='white', label='Detected Ball')
    axs[1,1].set_title('Overlay')


    fig.tight_layout()

    plt.show()

    





def validate(model, val_loader, criterion, device, writer=None, epoch=None):
    model.eval()
    val_loss = []
    tp = [0,0]
    fp = [0,0]
    tn = [0,0]
    fn = [0,0]
    max_dist = 10


    with torch.no_grad():
        for x, y, x_gt,y_gt,vis in val_loader:
            inputs, targets = x.to(device), y.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            print(f"Validation Batch Loss: {loss.item():.4f}")
            val_loss.append(loss.item())

            imgs_to_writer = torch.cat([targets, outputs], dim=0)  # Concatena inputs, targets e outputs para visualização

            writer.add_images('Target_Outputs/Val', imgs_to_writer, epoch, dataformats='NCHW')
            
            for img_idx in range(outputs.size(0)):
                img_to_show = outputs[img_idx]
                img_to_show = img_to_show.squeeze(0).cpu().numpy()  # (H, W)
               
                x, y = postprocess(img_to_show)
                gt_x, gt_y = x_gt[img_idx].item(), y_gt[img_idx].item()
                visibility = vis[img_idx].item()
            
                

                if x is not None and y is not None:
                    if visibility == 1:  # Ball is visible
                        dist = np.sqrt((x - gt_x) ** 2 + (y - gt_y) ** 2)
                        if dist <= max_dist:
                            tp[visibility] += 1  # True Positive
                        else:
                            fp[visibility] += 1  # False Positive
                    else:
                        fp[visibility] += 1  # False Positive (ball not visible but detected)
                else:
                    if visibility == 1:  # Ball is visible but not detected
                        fn[visibility] += 1  # False Negative
                    else:
                        tn[visibility] += 1  # True Negative (ball not visible and not detected)

        print(f"TP (Visible): {tp[1]}, FP (Visible): {fp[1]}, TN (Not Visible): {tn[0]}, FN (Visible): {fn[1]}")
        eps = 1e-6
        precision = tp[1] / (tp[1] + fp[1] + eps)        
        recall = tp[1] / (tp[1] + fn[1] + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)  
        
        avg_val_loss = sum(val_loss) / len(val_loss)


    return avg_val_loss, precision, recall, f1_score
            

if __name__ == "__main__":
    # Exemplo de uso
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    model = TrackNet().to(device)
    weight_path = "C:\\Users\\gears\\OneDrive\\Desktop\\Tracknet Goalball\\weights\\last_model.pth"

    model.load_state_dict(torch.load(weight_path, map_location=device))



    dataset = TracknetV2Dataset(frames_path="assets/dataset/frames_out", gts_path="assets/dataset/gts", labels_path="assets/dataset/labels", debug=True)
    x, y, x_pos, y_pos, vis = dataset[720*5*4]

    model.eval()
    with torch.no_grad():
        output = model(x.unsqueeze(0).to(device))
        output = output.squeeze(0).cpu().numpy()  # (C, H, W)
        postprocess_grid(y.squeeze(0), output[0], threshold=0.99)




