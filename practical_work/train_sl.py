import sys
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from tqdm.auto import tqdm
import wandb

from datasets import WildfirePredictionDataset
from transformations import RandomTransformation
from losses import AverageMeter, NTXentLoss, AutoassociativeLoss
from models import UNet, UNetClassifier


def save_checkpoint(
        checkpoint, is_best, path="checkpoint.pth.tar",
        best_path="checkpoint_best.pth.tar"):
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, best_path)


def train_epoch(
        epoch, model, data_loader, criterion, optimizer, config):
    # Set-up
    model.train()
    device = next(model.parameters()).device
    
    # Iterate over all batches
    for i, batch in enumerate(data_loader):
        # Load batch
        img, label = batch
        img = img.to(config.device)
        label = label.to(config.device)

        # Forward pass
        out = model(img)
        loss = criterion(out, label)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimisation step
        optimizer.step()

        # Logging
        loss_dict = {
            "loss": loss.item(),
        }
        log_dict = {
            "epoch": epoch
        }
        log_dict |= loss_dict
        wandb.log(log_dict, step=epoch)
        print(
            f"Epoch {epoch}: ["
            f"{i*len(img)}/{len(data_loader.dataset)}"
            f" ({100. * i / len(data_loader):.0f}%)]"
            f"{[f'{k} = {v:.6f}' for k, v in loss_dict.items()]}"
        )


def validation(
        epoch, model, data_loader, criterion, config):
    # Set-up
    model.eval()
    device = next(model.parameters()).device

    # Init measures
    avg_loss = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Load batch
            img, label = batch
            img = img.to(config.device)
            label = label.to(config.device)

            # Forward pass
            out = model(img)
            loss = criterion(out, label)

            # Update measures
            avg_loss.update(loss)

    # Logging
    log_dict = {
        "epoch": epoch,
        "validation_loss": avg_loss.avg
    }
    wandb.log(log_dict)
    print(
        f"Validation: {f"{[f'{k} = {v:.6f}' for k, v in log_dict.items()]}"}"
    )

    return avg_loss.avg


def train(
        model, train_loader, valid_loader, criterion, optimizer,
        lr_scheduler, config):
    # Configure wandb
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    best_loss = float("inf") # Best loss
    for epoch in tqdm(range(config.epochs)):
        print(f"#### EPOCH {epoch:4}/{config.epochs} ####")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Train for one epoch
        train_epoch(epoch, model, train_loader, criterion, optimizer, config)

        # Validation
        loss = validation(epoch, model, valid_loader, criterion, config)

        # Learning rate scheduler step
        lr_scheduler.step(loss)

        # Save model
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        save_checkpoint(
            checkpoint,
            is_best,
            os.path.join(config.save_path, "checkpoint.pth.tar"),
            os.path.join(config.save_path, "checkpoint_best.pth.tar")
        )


def make(config):
    # Model
    if config.model == "ResNet":
        # Create model
        model = resnet18(weights=ResNet18_Weights.DEFAULT)

        if config.encoder_id is not None:
            # Load SSL model
            model.fc = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            checkpoint = torch.load(f"train_res/{config.encoder_id}/checkpoint_best.pth.tar",
                weights_only=True, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])

        # Encoder freezing
        if config.encoder_freezing:
            for param in model.parameters():
                param.requires_grad = False

        # Modify model for SL
        model.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    elif config.model == "UNet":
        # Create model for SL
        model = UNetClassifier(classes=2, freeze_encoder=config.encoder_freezing)

        if config.encoder_id is not None:
            # Load SSL model
            unet_ssl = UNet()
            checkpoint = torch.load(f"train_res/{config.encoder_id}/checkpoint_best.pth.tar",
                weights_only=True, map_location=torch.device("cpu"))
            unet_ssl.load_state_dict(checkpoint["state_dict"])
            model.encoder = unet_ssl.encoder # Change encoder

        # Encoder freezing
        if config.encoder_freezing:
            for param in model.encoder.parameters():
                param.requires_grad = False
    else:
        raise ValueError("Invalid model")
    model = model.to(device)

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Training, validation and test data loaders
    loaders = WildfirePredictionDataset.get_dataloaders(transform=transform, batch_size=config.batch_size)

    # Criterion
    criterion = nn.CrossEntropyLoss() # No weight, classes are balanced

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    return model, *loaders, criterion, optimizer, lr_scheduler


def model_pipeline(config):
    # Link to wandb project
    with wandb.init(project="ROB313_PW", config=config):
        # Access config
        config = wandb.config
        print(config)

        # Make model, data and optimizater
        model, train_loader, valid_loader_1, valid_loader_2, test_loader, criterion, optimizer, lr_scheduler = make(config)
        print(model)

        # Train model
        train(model, valid_loader_2, valid_loader_1, criterion, optimizer, lr_scheduler, config)

    return model


if __name__ == "__main__":
    job_id = sys.argv[1]

    # Configure GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment configuration
    config_resnet = dict(
        job_id=job_id,
        model="ResNet",
        # encoder_id=285136,
        encoder_id=None,
        encoder_freezing=True,
        epochs=100,
        batch_size=128,
        learning_rate=1e-4,
        device=device,
        save_path=f"train_res/{job_id}"
    )

    config_unet = dict(
        job_id=job_id,
        model="UNet",
        # encoder_id=285138,
        encoder_id=None,
        encoder_freezing=False,
        epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        device=device,
        save_path=f"train_res/{job_id}"
    )
    
    model_pipeline(config_unet)