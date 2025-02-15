import sys
import os
import shutil
import logging

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
from models import UNet


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

    # Crete augmentation function
    augmentation = RandomTransformation((350,350))
    
    # Iterate over all batches
    for i, batch in enumerate(data_loader):
        # Load batch
        img, _ = batch
        img = img.to(config.device)

        # Apply augmentation function
        img_1 = augmentation(img)
        img_2 = augmentation(img)

        # Recreate batch (for positive pairs)
        augmented_img = torch.zeros(
                (img.shape[0]*2, img.shape[1], img.shape[2], img.shape[3]),
                device=config.device)
        for j in range(img.shape[0]):
            augmented_img[2*j] = img_1[j]
            augmented_img[2*j+1] = img_2[j]

        # Forward pass
        out = model(augmented_img)
        if config.model == "ResNet":
            loss, loss_dict = criterion(out)
        elif config.model == "UNet":
            loss, loss_dict = criterion(out["x_hat"], out["z_hat"], augmented_img)
        else:
            raise ValueError("Invalid model")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimisation step
        optimizer.step()

        # Logging
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
    
    return train_loss


def validation(
        epoch, model, data_loader, criterion, config):
    # Set-up
    model.eval()
    device = next(model.parameters()).device

    # Init measures
    avg_loss = AverageMeter()

    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Load batch
            img, _ = batch
            img = img.to(config.device)

            # Apply augmentation function
            img_1 = augmentation(img)
            img_2 = augmentation(img)

            # Recreate batch (for positive pairs)
            augmented_img = torch.zeros(
                (img.shape[0]*2, img.shape[1], img.shape[2], img.shape[3]),
                device=config.device)
            for i in range(img.shape[0]):
                augmented_img[2*i] = img_1[i]
                augmented_img[2*i+1] = img_2[i]

            # Forward pass
            out = model(augmented_img)
            if config.model == "ResNet":
                loss, loss_dict = criterion(out)
            elif config.model == "UNet":
                loss, loss_dict = criterion(out["x_hat"], out["z_hat"], augmented_img)
            else:
                raise ValueError("Invalid model")

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
        logging.info(f"#### EPOCH {epoch:4}/{config.epochs} ####")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Train for one epoch
        train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer, config)

        # Validation
        valid_loss = validation(epoch, model, valid_loader, criterion, config)
            
        logging.info(f"Train:      loss={train_loss:.8f}")
        logging.info(f"Validation: loss={valid_loss:.8f}")

        # Learning rate scheduler step
        lr_scheduler.step(valid_loss)

        # Save model
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": valid_loss,
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
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
    elif config.model == "UNet":
        model = UNet()
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
    if config.model == "ResNet":
        criterion = NTXentLoss(temperature=config.temperature)
    elif config.model == "UNet":
        criterion = AutoassociativeLoss(lmbda=config.lmbda,
                                        temperature=config.temperature)
    else:
        raise ValueError("Invalid model")

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
        train(model, train_loader, valid_loader_1, criterion, optimizer, lr_scheduler, config)

    return model


if __name__ == "__main__":
    job_id = sys.argv[1]

    # Configure GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment configuration
    config = dict(
        job_id=job_id,
        model="ResNet",
        # model="UNet",
        epochs=100,
        batch_size=128,
        # batch_size=16,
        learning_rate=1e-4,
        lmbda=0.5,
        temperature=0.1,
        device=device,
        save_path=f"train_res/{job_id}"
    )
    
    model_pipeline(config)