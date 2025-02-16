import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class WildfirePredictionDataset(Dataset):
    """
    Wildfire prediction dataset.
    Inspired by: https://github.com/Silectio/Projet-Computer-Vision/blob/main/loadDataset.py
    """
     # Label to class dictionary
    labels_dict = {
        0: "nowildfire",
        1: "wildfire"
    }
    
    def __init__(
            self, root_dir="/home/ids/fallemand-24/ROB313/data/wildfire-prediction-dataset",
            split="train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for cls_idx, cls_name in self.labels_dict.items():
            folder = os.path.join(self.root_dir, cls_name)
            for img_path in glob.glob(os.path.join(folder, "*.jpg")):
                self.image_paths.append(img_path)

                # Remove labels for train split!!!
                if split == "train":
                    self.labels.append(-1) 
                else:
                    self.labels.append(cls_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    @staticmethod
    def get_dataloaders(
            root_dir="/home/ids/fallemand-24/ROB313/data/wildfire-prediction-dataset",
            transform=None, batch_size=16, num_workers=2, shuffle=True,
            pin_memory=True):
        train_data = WildfirePredictionDataset(root_dir, "train", transform)
        valid_data = WildfirePredictionDataset(root_dir, "valid", transform)
        test_data = WildfirePredictionDataset(root_dir, "test", transform)

        # Split validation set into two subsets
        valid_data_1, valid_data_2 = torch.utils.data.random_split(valid_data, [0.5, 0.5])

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader_1 = DataLoader(valid_data_1, batch_size=batch_size, shuffle=shuffle)
        valid_loader_2 = DataLoader(valid_data_2, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

        return train_loader, valid_loader_1, valid_loader_2, test_loader