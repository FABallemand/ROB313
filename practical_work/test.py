import sys
import os
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary
from torcheval.metrics import (
    BinaryAccuracy, BinaryAUROC, BinaryConfusionMatrix, BinaryF1Score,
    BinaryPrecision, BinaryRecall)

import matplotlib.pyplot as plt
from PIL import Image

from datasets import WildfirePredictionDataset
from transformations import RandomTransformation
from models import UNet, UNetClassifier

def test_model(model, data_loader, device):

    metrics = {
        "accuracy": BinaryAccuracy(device=device),
        "precision": BinaryPrecision(device=device),
        "recall": BinaryRecall(device=device),
        "f1_score": BinaryF1Score(device=device),
        "confusion": BinaryConfusionMatrix(device=device),
    }

    with torch.no_grad():
        for batch in data_loader:
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            out = model(img)
            _, pred = torch.max(out, dim=1)
            
            for m in metrics.keys():
                metrics[m].update(pred, label)

    for m in metrics.keys():
        metrics[m] = metrics[m].compute()
        if isinstance(metrics[m], torch.Tensor):
            metrics[m] = metrics[m].cpu().numpy().tolist()

    return metrics

def test_resnet(sl_id, ssl_id, data_loader, device, save_path):
    print("==== ResNet ====")

    # Create models

    # ResNet SL
    # For baseline comparison
    resnet_sl = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_sl.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    checkpoint = torch.load(f"train_res/{sl_id}/checkpoint_best.pth.tar",
        weights_only=True, map_location=torch.device("cpu"))
    resnet_sl.load_state_dict(checkpoint["state_dict"])
    resnet_sl = resnet_sl.eval().to(device)

    # ResNet SSL
    resnet_ssl = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet_ssl.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    )
    checkpoint = torch.load(f"train_res/{ssl_id}/checkpoint_best.pth.tar",
        weights_only=True, map_location=torch.device("cpu"))
    resnet_ssl.load_state_dict(checkpoint["state_dict"])
    resnet_ssl = resnet_ssl.eval().to(device)

    sl_metrics = test_model(resnet_sl, data_loader, device)
    ssl_metrics = test_model(resnet_ssl, data_loader, device)

    with open(os.path.join(save_path, "resnet_sl_metrics.json"), "w") as f:
        json.dump(sl_metrics, f)
    with open(os.path.join(save_path, "resnet_ssl_metrics.json"), "w") as f:
        json.dump(ssl_metrics, f)

    print(f"SL  [{sl_id}]: {[f'{m} = {v}' for m, v in sl_metrics.items()]}")
    print(f"SSL [{ssl_id}]: {[f'{m} = {v}' for m, v in ssl_metrics.items()]}")


def test_unet(sl_id, ssl_id, data_loader, device, save_path):
    print("==== UNet ====")

    # Create models

    # UNet SL
    # For baseline comparison
    unet_sl = UNetClassifier()
    checkpoint = torch.load(f"train_res/{sl_id}/checkpoint_best.pth.tar",
        weights_only=True, map_location=torch.device("cpu"))
    unet_sl.load_state_dict(checkpoint["state_dict"])
    unet_sl = unet_sl.eval().to(device)

    # UNet SSL
    unet_ssl = UNetClassifier()
    checkpoint = torch.load(f"train_res/{ssl_id}/checkpoint_best.pth.tar",
        weights_only=True, map_location=torch.device("cpu"))
    unet_ssl.load_state_dict(checkpoint["state_dict"])
    unet_ssl = unet_ssl.eval().to(device)

    sl_metrics = test_model(unet_sl, data_loader, device)
    ssl_metrics = test_model(unet_ssl, data_loader, device)

    with open(os.path.join(save_path, "unet_sl_metrics.json"), "w") as f:
        json.dump(sl_metrics, f)
    with open(os.path.join(save_path, "unet_ssl_metrics.json"), "w") as f:
        json.dump(ssl_metrics, f)

    print(f"SL  [{sl_id}]: {[f'{m} = {v}' for m, v in sl_metrics.items()]}")
    print(f"SSL [{ssl_id}]: {[f'{m} = {v}' for m, v in ssl_metrics.items()]}")


def main(save_path):
    # Set-up
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test data loader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    loaders = WildfirePredictionDataset.get_dataloaders(transform=transform,
                                                        batch_size=16)
    test_loader = loaders[3]

    # Test ResNets
    resnet_sl_id = 285460
    resnet_ssl_id = 285444
    test_resnet(resnet_sl_id, resnet_ssl_id, test_loader, device, save_path)

    # Test UNets
    # unet_sl_id = 286618 # Encoder freezing
    unet_sl_id = 286619 # No encoder freezing
    unet_ssl_id = 286611
    test_unet(unet_sl_id, unet_ssl_id, test_loader, device, save_path)
    

if __name__ == "__main__":
    job_id = sys.argv[1]
    save_path=f"test_res/{job_id}"

    main(save_path)