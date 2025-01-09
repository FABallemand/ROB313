import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import VQVAE, UtilLSTM

def train_lstm(model, train_loader, vqvae, epochs=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss() # Latent code prediction loss
    
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            batch, _ = batch # Retrieve only images
            batch = batch.to(device)
            
            # Get latent codes from VQ-VAE
            with torch.no_grad():
                x_recon, z, quantized, loss = vqvae(batch)
            
            # Prepare sequences for LSTM (reshape into sequences of latent codes)
            seq_latents = quantized.view(-1, 256, 25).permute(0, 2, 1) # [batch_size, 256, 25]
            
            # Forward pass through LSTM
            optimizer.zero_grad()
            
            latent_output = model(seq_latents)
            
            # Compute loss between predicted latent codes and actual codes
            loss = criterion(latent_output, seq_latents)
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # Set random seed for reproducability and dataset splitting
    torch.manual_seed(42)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load Fashion MNIST dataset
    dataset = datasets.FashionMNIST("../data", download=True, transform=transform)

    # Define training, validation and test ratios
    train_ratio = 0.6
    validation_ratio = 0.2
    test_ratio = 0.2

    # Compute training, validation and test datasets sizes
    train_size = int(train_ratio * len(dataset))
    validation_size = int(validation_ratio * len(dataset))
    test_size = len(dataset) - train_size - validation_size

    # Create training, validation and test datasets
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

    # Create training, validation and test data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Instanciate VQ-VAE model
    vqvae = VQVAE()

    # Load weights
    id = 245458
    checkpoint = torch.load(f"train_res/{id}/checkpoint_best.pth.tar",
                            weights_only=True, map_location=torch.device("cpu"))
    vqvae.load_state_dict(checkpoint["state_dict"])
    vqvae = vqvae.eval().to(device)

    # Instanciate LSTM model
    lstm = UtilLSTM().to(device)

    # Train LSTM model
    train_lstm(lstm, train_loader, vqvae, epochs=5)

    # Save LSTM model
    torch.save(model.state_dict(), "lstm.pth.tar")