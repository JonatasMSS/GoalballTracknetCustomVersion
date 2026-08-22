
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms
import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

class TracknetV2Dataset(Dataset):
    def __init__(self, frames_path, gts_path, labels_path,original_width, original_height ,input_width=640, input_height=360, debug=False):
        self.frames_path = frames_path
        self.gts_path = gts_path
        self.labels_path = labels_path
        self.samples = []
        self.width = input_width
        self.height = input_height
        self.original_width = original_width
        self.original_height = original_height
        self.debug = debug


        self.transformation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.height, self.width)),
                transforms.ToTensor()
            ])

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.samples = []
        
        for partidas in sorted(os.listdir(self.frames_path)):
            clips_path = os.path.join(self.frames_path, partidas)
            for clip in sorted(os.listdir(clips_path)):
                frames = sorted(os.listdir(os.path.join(clips_path, clip)))
                for i in range(len(frames)):
                    if i >= 2:
                        path = os.path.join(self.frames_path, partidas, clip, frames[i])
                        path_prev = os.path.join(self.frames_path, partidas, clip, frames[i-1])
                        path_preprev = os.path.join(self.frames_path, partidas, clip, frames[i-2])


                        path_gt = os.path.join(self.gts_path, partidas, clip, frames[i])
                        path_gt_prev = os.path.join(self.gts_path, partidas, clip, frames[i-1])
                        path_gt_preprev = os.path.join(self.gts_path, partidas, clip, frames[i-2])


                        

                        label_path = os.path.join(self.labels_path, partidas, clip, frames[i]).replace('.jpg', '.txt')
                        label_path_prev = os.path.join(self.labels_path, partidas, clip, frames[i-1]).replace('.jpg', '.txt')
                        label_path_preprev = os.path.join(self.labels_path, partidas, clip, frames[i-2]).replace('.jpg', '.txt')
                        
                        self.samples.append((path, path_prev, path_preprev, path_gt, path_gt_prev, path_gt_preprev, label_path, label_path_prev, label_path_preprev))
                

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, path_prev, path_preprev, path_gt, path_gt_prev, path_gt_preprev, label_path, label_path_prev, label_path_preprev = self.samples[idx]

        x = self.get_input(path, path_prev, path_preprev)
        y = self.get_output(path_gt, path_gt_prev, path_gt_preprev)
        labels = self.get_labels(label_path, label_path_prev, label_path_preprev)


        return x, y, labels

    def show_sample(self,idx):
        x, y, labels = self.__getitem__(idx)
        x = self.denormalize_9ch(x)

        x1, x2, x3 = torch.chunk(x, 3, dim=0)
        y1, y2, y3 = torch.chunk(y, 3, dim=0)

        data = [x1, x2, x3, y1, y2, y3]

        plt.figure(figsize=(12, 8))

        for i, img in enumerate(data):

            plt.subplot(2, 3, i + 1)
            plt.axis("off")

            if i < 3:
                # [3,H,W] -> [H,W,3]
                img = img.permute(1, 2, 0).cpu().numpy()

                # garantir intervalo [0,1]
                img = np.clip(img, 0, 1)

                # cv2.circle trabalha melhor aqui com uma cópia
                img = img.copy()

                x_pos = int(labels[i][0])
                y_pos = int(labels[i][1])

                cv2.circle(
                    img,
                    (x_pos, y_pos),
                    radius=5,
                    color=(0, 1, 0),
                    thickness=-1
                )

                plt.imshow(img)
                plt.title(f"Input Frame {i + 1}")

            else:
                # GT: [1,H,W] -> [H,W]
                img = img.squeeze(0).cpu().numpy()

                plt.imshow(img, cmap="gray")
                plt.title(f"Ground Truth Frame {i - 2}")

        plt.tight_layout()
        plt.show()
    def get_output(self, path_gt, path_gt_prev, path_gt_preprev):
        img = cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        img_prev = cv2.imread(path_gt_prev, cv2.IMREAD_GRAYSCALE)
        img_preprev = cv2.imread(path_gt_preprev, cv2.IMREAD_GRAYSCALE)

        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
        img_prev = torch.from_numpy(img_prev).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]
        img_preprev = torch.from_numpy(img_preprev).unsqueeze(0).float() / 255.0  # Normalize to [0, 1]

        y = torch.cat([img_preprev, img_prev, img], dim=0)  # (3, H, W)

        return y

    def denormalize_9ch(self, tensor):
        """
        tensor: [9, H, W]
        retorna: [9, H, W]
        """

        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=tensor.device,
            dtype=tensor.dtype
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=tensor.device,
            dtype=tensor.dtype
        ).view(3, 1, 1)

        img1, img2, img3 = torch.chunk(tensor, 3, dim=0)

        img1 = img1 * std + mean
        img2 = img2 * std + mean
        img3 = img3 * std + mean

        return torch.cat([img1, img2, img3], dim=0)
    def _get_label_data(self, label_path):
        if not os.path.exists(label_path):
            return 0, 0, 0  # Return default values if the label file doesn't exist

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if len(lines) == 0:
            return 0, 0, 0  # Return default values if the label file is empty

        lines = lines[0].strip().split()

        x_pos = (float(lines[0]) * self.width)/ self.original_width 
        y_pos = (float(lines[1]) * self.height)/ self.original_height 
        vis = int(lines[2]) 


        return x_pos, y_pos, vis

    def get_labels(self, label_path,label_path_prev,label_path_preprev):
        x_pos, y_pos, vis = self._get_label_data(label_path)
        x_pos_prev, y_pos_prev, vis_prev = self._get_label_data(label_path_prev)
        x_pos_preprev, y_pos_preprev, vis_preprev = self._get_label_data(label_path_preprev)

        return [(x_pos_preprev, y_pos_preprev, vis_preprev),(x_pos_prev, y_pos_prev, vis_prev),(x_pos, y_pos, vis)]

        
    def get_input(self, path, path_prev, path_preprev):

        img = cv2.imread(path)
        img_prev = cv2.imread(path_prev)
        img_preprev = cv2.imread(path_preprev)

            # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_prev = cv2.cvtColor(img_prev, cv2.COLOR_BGR2RGB)
        img_preprev = cv2.cvtColor(img_preprev, cv2.COLOR_BGR2RGB)

        img = self.transformation(img)
        img_prev = self.transformation(img_prev)
        img_preprev = self.transformation(img_preprev)

        img = self.normalize(img)
        img_prev = self.normalize(img_prev)
        img_preprev = self.normalize(img_preprev)


        imgs = torch.cat([img_preprev, img_prev, img], dim=0)  # (9, H, W)
        
        return imgs

        