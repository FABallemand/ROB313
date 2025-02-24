import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class WildfireDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        classes = ["wildfire", "nowildfire"]
        for cls_idx, cls_name in enumerate(classes):
            folder = os.path.join(self.root_dir, cls_name)
            for img_path in glob.glob(os.path.join(folder, "*.jpg")):
                self.image_paths.append(img_path)

                # On enlève bien les lables pour le train split !!!
                if split == 'train':
                    self.labels.append(0) 
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

def get_dataloaders(root_dir="dataset", transform=None, batch_size=16):
    train_data = WildfireDataset(root_dir, 'train', transform)
    valid_data = WildfireDataset(root_dir, 'valid', transform)
    test_data = WildfireDataset(root_dir, 'test', transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
