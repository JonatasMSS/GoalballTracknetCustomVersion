
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms
import os
import re
import cv2
import numpy as np

import matplotlib.pyplot as plt

class TracknetV2Dataset(Dataset):
    def __init__(self, frames_path, gts_path, labels_path,original_width, original_height ,input_width=640, input_height=360, debug=False):
        if not np.isfinite([original_width, original_height]).all() or original_width <= 0 or original_height <= 0:
            raise ValueError("original_width and original_height must be finite and greater than zero")
        if (
            not np.isfinite([input_width, input_height]).all()
            or input_width <= 0 or input_height <= 0
            or input_width % 8 != 0 or input_height % 8 != 0
        ):
            raise ValueError("input_width and input_height must be positive multiples of 8")
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
                frames = sorted(
                    os.listdir(os.path.join(clips_path, clip)),
                    key=lambda name: (
                        [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', name)],
                        name,
                    ),
                )
                if len(frames) < 3:
                    continue
                gt_clip_path = os.path.join(self.gts_path, partidas, clip)
                gt_index = {}
                for filename in sorted(os.listdir(gt_clip_path)):
                    stem, extension = os.path.splitext(filename)
                    if extension.lower() in {'.jpg', '.jpeg', '.png'}:
                        gt_index.setdefault(stem, []).append(os.path.join(gt_clip_path, filename))
                gt_paths = []
                for filename in frames:
                    exact_path = os.path.join(gt_clip_path, filename)
                    candidates = gt_index.get(os.path.splitext(filename)[0], [])
                    if os.path.isfile(exact_path):
                        gt_paths.append(exact_path)
                    elif len(candidates) == 1:
                        gt_paths.append(candidates[0])
                    elif not candidates:
                        frame_path = os.path.join(clips_path, clip, filename)
                        raise FileNotFoundError(f"Missing ground truth for frame {frame_path}; expected {exact_path}")
                    else:
                        frame_path = os.path.join(clips_path, clip, filename)
                        raise ValueError(f"Ambiguous ground truth for frame {frame_path}: {', '.join(candidates)}")
                for i in range(len(frames)):
                    if i >= 2:
                        path = os.path.join(self.frames_path, partidas, clip, frames[i])
                        path_prev = os.path.join(self.frames_path, partidas, clip, frames[i-1])
                        path_preprev = os.path.join(self.frames_path, partidas, clip, frames[i-2])


                        path_gt = gt_paths[i]
                        path_gt_prev = gt_paths[i-1]
                        path_gt_preprev = gt_paths[i-2]


                        

                        label_path = os.path.splitext(os.path.join(self.labels_path, partidas, clip, frames[i]))[0] + '.txt'
                        label_path_prev = os.path.splitext(os.path.join(self.labels_path, partidas, clip, frames[i-1]))[0] + '.txt'
                        label_path_preprev = os.path.splitext(os.path.join(self.labels_path, partidas, clip, frames[i-2]))[0] + '.txt'
                        
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
        heatmaps = []
        for gt_path in (path_gt_preprev, path_gt_prev, path_gt):
            image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Unable to read ground truth: {gt_path}")
            if image.shape != (self.height, self.width):
                raise ValueError(
                    f"Ground truth {gt_path} has shape {image.shape}; expected {(self.height, self.width)}"
                )
            heatmaps.append(torch.from_numpy(image).unsqueeze(0).float() / 255.0)
        return torch.cat(heatmaps, dim=0)

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
        with open(label_path, 'r') as f:
            fields = f.readline().split()

        if len(fields) != 3:
            raise ValueError(f"Invalid label {label_path}: expected 'x y vis' on the first line")
        try:
            x_pos, y_pos = float(fields[0]), float(fields[1])
            vis = int(fields[2])
        except ValueError as exc:
            raise ValueError(f"Invalid label {label_path}: x/y must be numbers and vis an integer") from exc
        if not np.isfinite([x_pos, y_pos]).all():
            raise ValueError(f"Invalid label {label_path}: x/y must be finite")

        x_pos = (x_pos * self.width) / self.original_width
        y_pos = (y_pos * self.height) / self.original_height

        return x_pos, y_pos, vis

    def get_labels(self, label_path,label_path_prev,label_path_preprev):
        x_pos, y_pos, vis = self._get_label_data(label_path)
        x_pos_prev, y_pos_prev, vis_prev = self._get_label_data(label_path_prev)
        x_pos_preprev, y_pos_preprev, vis_preprev = self._get_label_data(label_path_preprev)

        return [(x_pos_preprev, y_pos_preprev, vis_preprev),(x_pos_prev, y_pos_prev, vis_prev),(x_pos, y_pos, vis)]

        
    def get_input(self, path, path_prev, path_preprev):
        frames = []
        for frame_path in (path_preprev, path_prev, path):
            image = cv2.imread(frame_path)
            if image is None:
                raise ValueError(f"Unable to read frame: {frame_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(self.normalize(self.transformation(image)))
        return torch.cat(frames, dim=0)
