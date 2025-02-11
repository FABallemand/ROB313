import os
import psutil
import logging

import torch

from datasets import WildfirePredictionDataset
from models import UNet


def save_checkpoint(
        checkpoint, is_best, path="checkpoint.pth.tar",
        best_path="checkpoint_best.pth.tar"):
    torch.save(checkpoint, path)
    if is_best:
        shutil.copyfile(path, best_path)


def train_epoch(
        epoch, model, train_loader, criterion, optimizer, device, results_dir,
        debug=False):

    # Enable training
    model.train(True)

    # Initialise loss
    train_loss = 0.0
    
    # Pass over all batches
    for i, batch in enumerate(train_loader):

        # Load and prepare batch
        img, _ = batch

        # Split batch
        h = int(batch_img.shape[0] / 2)
        img_1, img_3 = batch_img[:h,], batch_img[h:,]

        # Apply transofrmation to first split
        img_2 = img_1 # TODO

        # Zero gradient
        optimizer.zero_grad()

        # Reconstruction on batches
        encoded_1, decoded_1 = model(img_1)
        encoded_2, decoded_2 = model(img_2)
        encoded_3, decoded_3 = model(img_3)

        # Compute loss
        loss = contrastive_reconstruction_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                               p_decoded_x_1, p_rgb_1, reconstruction_loss_function) # TODO

        # Compute gradient loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Update losses
        train_loss += loss.item()

        # Log
        print(
            f"Epoch {epoch}: ["
            f"{i*len(x)}/{len(train_loader.dataset)}"
            f" ({100. * i / len(train_loader):.0f}%)]"
            f"{[f'{k} = {v:.6f}' for k, v in loss_dict.items()]}"
        )

    # Compute training loss
    # TODO Use average meter
    train_loss /= (i + 1) # Average loss over all batches of the epoch
    
    return train_loss


def evaluate(
        model, loader, reconstruction_loss_function, epoch, device,
        results_dir):

    # Initialise losses
    validation_loss = 0.0

    # Freeze the model
    model.eval()
    with torch.no_grad():

        # Iterate over batches
        for i, batch in enumerate(loader):
            
            # Load and prepare batch
            p_data_1, p_data_2, n_data = batch
            p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
            p_rgb_1 = p_rgb_1.to(device)
            # p_depth_1 = p_depth_1.to(device)
            # p_mask_1 = p_mask_1.to(device)
            # p_loc_x_1 = p_loc_x_1.to(device)
            # p_loc_y_1 = p_loc_y_1.to(device)
            p_label_1 = p_label_1.to(device)
            p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
            p_rgb_2 = p_rgb_2.to(device)
            # p_depth_2 = p_depth_2.to(device)
            # p_mask_2 = p_mask_2.to(device)
            # p_loc_x_2 = p_loc_x_2.to(device)
            # p_loc_y_2 = p_loc_y_2.to(device)
            p_label_2 = p_label_2.to(device)
            n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
            n_rgb = n_rgb.to(device)
            # n_depth = n_depth.to(device)
            # n_mask = n_mask.to(device)
            # n_loc_x = n_loc_x.to(device)
            # n_loc_y = n_loc_y.to(device)
            n_label = n_label.to(device)

            # Make predictions for batch
            p_encoded_x_1, p_decoded_x_1 = model(p_rgb_1)
            p_encoded_x_2, p_decoded_x_2 = model(p_rgb_2)
            n_encoded_x, n_decoded_x = model(n_rgb)

            # Compute loss
            loss = contrastive_reconstruction_loss(p_encoded_x_1, p_encoded_x_2, n_encoded_x,
                                                   p_decoded_x_1, p_rgb_1, reconstruction_loss_function)

            # Update batch loss
            validation_loss += loss.item()

    # Compute validation loss
    validation_loss /= (i + 1) # Average loss over all batches

    return validation_loss


def train(
        model, train_loader, validation_loader, criterion,
        optimizer, epochs, lr_scheduler, device=torch.device("cpu"),
        results_dir="test", debug=False):

    # Losses
    train_losses = []
    validation_losses = []
    
    best_loss = float("inf") # Best loss
    for epoch in range(1, epochs + 1):
        logging.info(f"#### EPOCH {epoch:4}/{epochs} ####")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, loss_function, optimizer, epoch, device, results_dir)
        train_losses.append(train_loss)

        # Evaluate model
        validation_loss = evaluate(model, validation_loader, loss_function, epoch, device, results_dir)
        validation_losses.append(validation_loss)
            
        logging.info(f"Train:      loss={train_loss:.8f}")
        logging.info(f"Validation: loss={validation_loss:.8f}")

        # Learning rate scheduler step
        lr_scheduler.step(validation_loss)

        # Save model
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": validation_loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        }
        save_checkpoint(
            checkpoint,
            is_best,
            os.path.join(config.save_path, "checkpoint.pth.tar"),
            os.path.join(config.save_path, "checkpoint_best.pth.tar")
        )

    return train_losses, validation_losses


def test(model, test_loader, device=torch.device("cpu")):

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            
            # Load and prepare batch
            p_data_1, p_data_2, n_data = batch
            p_rgb_1, p_depth_1, p_mask_1, p_loc_x_1, p_loc_y_1, p_label_1 = p_data_1
            p_rgb_1 = p_rgb_1.to(device)
            # p_depth_1 = p_depth_1.to(device)
            # p_mask_1 = p_mask_1.to(device)
            # p_loc_x_1 = p_loc_x_1.to(device)
            # p_loc_y_1 = p_loc_y_1.to(device)
            p_label_1 = p_label_1.to(device)
            p_rgb_2, p_depth_2, p_mask_2, p_loc_x_2, p_loc_y_2, p_label_2 = p_data_2
            p_rgb_2 = p_rgb_2.to(device)
            # p_depth_2 = p_depth_2.to(device)
            # p_mask_2 = p_mask_2.to(device)
            # p_loc_x_2 = p_loc_x_2.to(device)
            # p_loc_y_2 = p_loc_y_2.to(device)
            p_label_2 = p_label_2.to(device)
            n_rgb, n_depth, n_mask, n_loc_x, n_loc_y, n_label = n_data
            n_rgb = n_rgb.to(device)
            # n_depth = n_depth.to(device)
            # n_mask = n_mask.to(device)
            # n_loc_x = n_loc_x.to(device)
            # n_loc_y = n_loc_y.to(device)
            n_label = n_label.to(device)
            
            # Make predictions for batch
            p_encoded_x_1, p_predicted_label_1 = model(p_rgb_1)
            p_encoded_x_2, p_predicted_label_2 = model(p_rgb_2)
            n_encoded_x, n_predicted_label = model(n_rgb)

            # Save encoded features and labels
            encoded_features.append(p_encoded_x_1)
            labels.append(p_label_1)
            encoded_features.append(p_encoded_x_2)
            labels.append(p_label_2)
            encoded_features.append(n_encoded_x)
            labels.append(n_label)


def main(results_dir):

    # Model
    model = UNet()

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Training, validation and test data loaders
    loaders = WildfirePredictionDataset.get_dataloaders(transform=transform, batch_size=16)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

     # Train model
    train(teacher_model, student_model, train_loader, validation_loader, criterion, optimizer, lr_scheduler, config)

    # Test model
    test(student_model, test_loader, config)


if __name__ == "__main__":
    job_id = sys.argv[1]
    results_dir = f"train_res/{job_id}"
    main(results_dir)