
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

class TracknetV2Dataset(Dataset):
    def __init__(self, frames_path = './dataset/frames_out',gts_path="./dataset/gts",labels_path="./dataset/labels", input_height=360, input_width=640, debug = False):
        self.frames_path = frames_path
        self.gts_path = gts_path
        self.labels_path = labels_path
        self.samples = []
        self.width = input_width
        self.height = input_height
        self.debug = debug

        self.transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.samples = []
        
        for partidas in os.listdir(self.frames_path):
            clips_path = os.path.join(self.frames_path, partidas)
            for clip in os.listdir(clips_path):
                frames = os.listdir(os.path.join(clips_path, clip))
                for i in range(len(frames)):
                    if i >= 2:
                        path = os.path.join(self.frames_path, partidas, clip, frames[i])
                        path_prev = os.path.join(self.frames_path, partidas, clip, frames[i-1])
                        path_preprev = os.path.join(self.frames_path, partidas, clip, frames[i-2])


                        path_gt = os.path.join(self.gts_path, partidas, clip, frames[i]).replace('.jpg', '.png')
                        

                        label_path = os.path.join(self.labels_path, partidas, clip, frames[i]).replace('.jpg', '.txt')
                        
                        self.samples.append((path, path_prev, path_preprev, path_gt, label_path))
                
          
    def get_label_data(self, label_path):
        if self.debug:
            print(f"Reading label from: {label_path}")
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if line:
                x_str, y_str, visibility = line.split(' ')
                return int(x_str), int(y_str), int(visibility)
            else:
                return None, None, None
    
    def display_samples(self,idx):
        path, _, _, path_gt, label_path = self.samples[idx]
        img = cv2.imread(path)
        img_gt = cv2.imread(path_gt)

        x,y, vis = self.get_label_data(label_path)
        if vis != 0:
            cv2.circle(img, (x, y), radius=20, color=(0, 255, 0), thickness=-1)
        else:
            print(f"Ball not visible in sample {idx}")
        img = self.transformation(img)
        img_gt = self.transformation(img_gt)
        grid = make_grid([img, img_gt], nrow=2)
        return grid
    
        


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, label_path = self.samples[idx]

        x = self.get_input(path, path_prev, path_preprev)
        y = self.get_output(path_gt)
        x_pos,y_pos,vis = self.get_label_data(label_path)
        
        if self.debug:
            
            max_value= torch.max(y).item()
            min_value = torch.min(y).item()

            print(f"Len of dataset: {len(self.samples)}")
            print(f"Sample {idx} - Input shape: {x.shape}, Output shape: {y.shape}, Visibility: {vis}, Max value in Y: {max_value}, Min value in Y: {min_value}")
            

        return x, y, x_pos, y_pos, vis
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        img = self.transformation(img)

        return img
        
    def get_input(self, path, path_prev, path_preprev):

        img = cv2.imread(path)
        img_prev = cv2.imread(path_prev)
        img_preprev = cv2.imread(path_preprev)

        img = self.transformation(img)
        img_prev = self.transformation(img_prev)
        img_preprev = self.transformation(img_preprev)

        img = self.normalize(img)
        img_prev = self.normalize(img_prev)
        img_preprev = self.normalize(img_preprev)


        imgs = torch.cat([img_preprev, img_prev, img], dim=0)  # (9, H, W)
        
        return imgs

        
    


if __name__ == "__main__":

    IDX = 0

    dataset = TracknetV2Dataset(frames_path="assets/dataset/frames_out", gts_path="assets/dataset/gts", labels_path="assets/dataset/labels", debug=True)

    x, y, x_pos, y_pos, vis = dataset[IDX]
    
    grid = dataset.display_samples(IDX)
    plt.imshow(grid.permute(1, 2, 0)) 
    plt.show()