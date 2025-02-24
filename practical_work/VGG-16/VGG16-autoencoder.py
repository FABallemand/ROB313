import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from loadDataset import get_dataloaders
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VGG16Encoder(nn.Module):
    def __init__(self):
        super(VGG16Encoder, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children())[:16])
    def forward(self, x):
        return self.features(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 3, 1, 1)
        )
    def forward(self, x):
        return self.decoder(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = VGG16Encoder()
        self.decoder = Decoder()
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

def train(train_loader, model,device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_loader, valid_loader, test_loader = get_dataloaders(transform=transform, batch_size=32)
    model = Autoencoder().to(device)
    #on affiche la taille du latent space
    for images, _ in train_loader:
        latentSize = Autoencoder().encoder(images).shape
        print(latentSize)
        break
    train(train_loader, model, device)

    torch.save(model.state_dict(), 'models/VGG16-Autoencoder_autoencoder.pth')
    model.encoder.load_state_dict(torch.load('models/VGG16-Autoencoder_autoencoder.pth'))
    
    # CLASSIFIER === 
    
    class Classifier(nn.Module):
        def __init__(self, encoder):
            super(Classifier, self).__init__()
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.fc = nn.Linear(256 * 14 * 14, 2)

        def forward(self, x):
            with torch.no_grad():
                x = self.encoder(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    def train_classifier(valid_loader, model, device):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
        model.train()
        for epoch in range(5):
            total_loss = 0
            for images, labels in tqdm(valid_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(valid_loader)}")

    # Assuming valid_loader has labels for classification
    classifier = Classifier(model.encoder).to(device)
    train_classifier(valid_loader, classifier, device)
    #on sauvegarde le modèle
    torch.save(classifier.state_dict(), 'models/VGG16-Autoencoder_classifier.pth')