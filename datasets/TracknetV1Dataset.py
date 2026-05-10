
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

class TracknetV1Dataset(Dataset):
    def __init__(self, frames_path = './dataset/frames_out',gts_path="./dataset/gts",labels_path="./dataset/labels", input_height=360, input_width=640):
        self.frames_path = frames_path
        self.gts_path = gts_path
        self.labels_path = labels_path
        self.samples = []
        self.width = input_width
        self.height = input_height

        self.transformation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.height, self.width)),
            transforms.ToTensor()
        ])

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

        return x, y, x_pos, y_pos, vis
    
    def get_output(self, path_gt):
        img = cv2.imread(path_gt)
        img = cv2.resize(img, (self.width, self.height))
        img = img[:, :, 0]
        img = np.reshape(img, (self.width * self.height))
        return img
        
    def get_input(self, path, path_prev, path_preprev):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))

        img_prev = cv2.imread(path_prev)
        img_prev = cv2.resize(img_prev, (self.width, self.height))
        
        img_preprev = cv2.imread(path_preprev)
        img_preprev = cv2.resize(img_preprev, (self.width, self.height))
        
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0

        imgs = np.rollaxis(imgs, 2, 0)
        return imgs

        
    


if __name__ == "__main__":

    IDX = 11

    dataset = TracknetV1Dataset(frames_path="assets/dataset/frames_out", gts_path="assets/dataset/gts", labels_path="assets/dataset/labels")
    print(len(dataset))
    x, y, x_pos, y_pos, vis = dataset[IDX]
    print(f'X shape: {x.shape}, Y shape: {y.shape}, Max value in Y: {np.max(y)}, Min value in Y: {np.min(y)}')

    grid = dataset.display_samples(IDX)
    plt.imshow(grid.permute(1, 2, 0)) 
    plt.show()