import os
import sys
import shutil
from datetime import datetime
import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import VQVAE

def save_checkpoint(
        state, is_best, filename="checkpoint.pth.tar",
        best_filename="checkpoint_best.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)

def train_one_epoch(model, data_loader, optimizer, device, epoch, results_dir):
    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0

    # Iterate over batches
    i = 0
    for i, batch in enumerate(data_loader):
        # Load batch
        x, _ = batch
        x = x.to(device)

        # Zero gradient
        optimizer.zero_grad()

        # Make predictions for batch
        x_recon, _, _, vq_loss = model(x)

        # Compute loss
        loss_dict = model.loss_function(x_recon, x, vq_loss)
        loss = loss_dict["loss"]

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        if i % 10 == 0:
            # Batch loss
            logging.info("\tBatch %s/%s: loss=%s | reconstruction_loss=%s | vq_loss=%s",
                         f"{i:8}", f"{len(data_loader)}", f"{loss_dict['loss']:.4f}",
                         f"{loss_dict['reconstruction_loss']:.4f}",
                         f"{loss_dict['vq_loss']:.4f}")

    # Compute training loss
    train_loss /= (i + 1) # Average loss over all batches of the epoch

    return train_loss


def evaluate(model, data_loader, device, results_dir):
    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():
        # Iterate over batches
        i = 0
        for i, batch in enumerate(data_loader):
            # Load batch
            x, _ = batch # Retrieve only images
            x = x.to(device)

            # Make predictions for batch
            x_recon, _, _, vq_loss = model(x)

            # Compute loss
            loss_dict = model.loss_function(x_recon, x, vq_loss)
            loss = loss_dict["loss"]

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation loss
    validation_loss /= (i + 1) # Average loss over all batches

    return validation_loss


def train(
        model, train_data_loader, validation_data_loader, epochs,
        learning_rate, device=torch.device("cpu"), results_dir="test"):
    # Initialise
    train_losses = []
    validation_losses = []
    best_validation_loss = float("inf")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(1, epochs + 1):
        logging.info("#### EPOCH %s/%s ####", f"{epoch:4}", epochs)

        # Train for one epoch
        train_loss = train_one_epoch(model, train_data_loader, optimizer, device, epoch, results_dir)
        train_losses.append(train_loss)

        # Evaluate model
        validation_loss = evaluate(model, validation_data_loader, device, results_dir)
        validation_losses.append(validation_loss)

        # Save model
        is_best = validation_loss < best_validation_loss
        best_validation_loss = min(validation_loss, best_validation_loss)
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "loss": validation_loss,
                "optimizer": optimizer.state_dict()
            },
            is_best,
            results_dir + "/checkpoint.pth.tar",
            results_dir + "/checkpoint_best.pth.tar"
        )

        logging.info("Train:      loss=%s", f"{train_loss:.8f}")
        logging.info("Validation: loss=%s", f"{validation_loss:.8f}")

    return train_losses, validation_losses


def test(model, data_loader, device=torch.device("cpu")):
    # Initialise losses
    test_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():
        # Iterate over batches
        i = 0
        for i, batch in enumerate(data_loader):
            # Load batch
            x, _ = batch # Retrieve only images
            x = x.to(device)

            # Make predictions for batch
            x_recon, _, _, vq_loss = model(x)

            # Compute loss
            loss_dict = model.loss_function(x_recon, x, vq_loss)
            loss = loss_dict["loss"]

            # Update batch loss
            test_loss += loss.item()

    # Compute validation loss
    test_loss /= (i + 1) # Average loss over all batches

    logging.info("\tTest: loss=%s | reconstruction_loss=%s | vq_loss=%s",
                 f"{loss_dict['loss']:.4f}",
                 f"{loss_dict['reconstruction_loss']:.4f}",
                 f"{loss_dict['vq_loss']:.4f}")

    return test_loss


def main(*args, **kwargs):
    # Retrieve arguments
    results_dir = args[0][0]

    # Configure logging
    format_ = "%(asctime)s %(message)s"
    logging.basicConfig(filename=os.path.join(results_dir, "training.log"),
                        level=logging.DEBUG, format=format_)
    
    # Save training time start
    start_timestamp = datetime.now()
    logging.info(start_timestamp.strftime("%Y%m%d_%H%M%S"))

    # Begin set-up
    logging.info("#### Set-Up ####")

    # Set-up PyTorch
    seed = 42
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("seed = %s", seed)
    logging.info("device = %s", device)

    # Dataset parameters
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Training parameters
    checkpoint = None # Last checkpoint to load

    batch_size = 128  # Batch size
    shuffle = True    # Shuffle
    drop_last = False # Drop last batch
    num_workers = 2   # Number of prpocesses
    pin_memory = True # Memory pinning

    epochs = 250000      # Number of epochs
    learning_rate = 2e-4 # Learning rates

    debug = False # Debug flag

    # Datasets
    logging.info("#### Datasets ####")

    # Load Fashion MNIST dataset
    dataset = datasets.FashionMNIST("../data", download=False, transform=transform)

    # Define training, validation and test ratios
    train_ratio = 0.6
    validation_ratio = 0.2
    # test_ratio = 0.2

    # Compute training, validation and test datasets sizes
    train_size = int(train_ratio * len(dataset))
    validation_size = int(validation_ratio * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    # Create training, validation and test datasets
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, [train_size, validation_size, test_size])

    # Training data loaders
    logging.info("#### Data Loaders ####")

    logging.info("batch_size = %s", batch_size)
    logging.info("shuffle = %s", shuffle)
    logging.info("drop_last = %s", drop_last)
    logging.info("num_workers = %s", num_workers)
    logging.info("pin_memory = %s", pin_memory)

    # Create training, validation and test data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_data_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Neural network
    logging.info("#### Model ####")

    logging.info("checkpoint = %s", checkpoint)
    logging.info("epochs = %s", epochs)
    logging.info("learning_rate = %s", learning_rate)
    logging.info("debug = %s", debug)

    # Create model
    model = VQVAE()

    # Load last checkpoint if specified
    if checkpoint is not None and os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint))

    # Load model to device
    model = model.to(device)

    # Print model
    logging.info(model)

    # Training
    logging.info("#### Training ####")

    # Train model
    train_loss, val_loss = train(model, train_data_loader,
                                 validation_data_loader, epochs, learning_rate,
                                 device, results_dir)

    # Save training time stop
    stop_timestamp = datetime.now()
    logging.info(stop_timestamp.strftime("%Y%m%d_%H%M%S"))

    # Testing
    logging.info("#### Testing ####")

    # Test model
    test_loss = test(model, test_data_loader, device)

    # End training
    logging.info("#### End ####")


if __name__ == "__main__":
    main(sys.argv[1:])
