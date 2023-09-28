import os
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
import pandas as pd

#
# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __len__(self):
#         return len(self.img_labels)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
#

class PandaDataset(Dataset):
    def __init__(self, data_dir, img_paths, labels, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.img_paths = img_paths
        self.labels = torch.Tensor(labels.astype(int))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.img_paths.iloc[int(idx)])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = image.transpose(2, 0, 1).astype('float32')
        label = self.labels[int(idx)]
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


