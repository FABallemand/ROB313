import torch
from torchvision import transforms
from loadDataset import get_dataloaders
from sklearn.mixture import GaussianMixture
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

class GMMClassifier:
    def __init__(self, n_components=2):
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    def fit(self, data):
        self.gmm.fit(data)

    def predict(self, data):
        return self.gmm.predict(data)

    def save(self, filepath):
        joblib.dump(self.gmm, filepath)

    def load(self, filepath):
        self.gmm = joblib.load(filepath)

def extract_features(dataloader, model, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            outputs = model.forward(images)
            features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
    return np.vstack(features)

def compute_accuracy(predictions, dataloader):
    correct = 0
    total = 0
    for i, (_, labels) in enumerate(dataloader):
        correct += (predictions[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] == labels.numpy()).sum()
        total += labels.size(0)
    return correct / total

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_loader, valid_loader, test_loader = get_dataloaders(transform=transform, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = vgg16(weights=VGG16_Weights.DEFAULT)

    # train_features = extract_features(train_loader, autoencoder, device)
    valid_features = extract_features(valid_loader, autoencoder, device)
    test_features = extract_features(test_loader, autoencoder, device)

    # gmm = GMMClassifier(n_components=2)
    # gmm.fit(train_features)

    # # Save the trained GMM model
    # gmm.save('gmm_model.pkl')

    gmm = GMMClassifier()
    gmm.load('gmm_model.pkl')
    
    # train_preds = gmm.predict(train_features)
    valid_preds = gmm.predict(valid_features)
    test_preds = gmm.predict(test_features)

    # print("Train Predictions:", train_preds)
    print("Validation Predictions:", valid_preds)
    print("Test Predictions:", test_preds)
    
    valid_accuracy = compute_accuracy(valid_preds, valid_loader)
    test_accuracy = compute_accuracy(test_preds, test_loader)

    print("Validation Accuracy:", valid_accuracy)
    print("Test Accuracy:", test_accuracy)

    confusion_matrix = np.zeros((2, 2))
    valid_preds = valid_preds[:len(valid_loader.dataset)]  # Ensure valid_preds length matches dataset length

    for i, (_, labels) in enumerate(valid_loader):
        for j in range(len(labels)):
            confusion_matrix[valid_preds[i * valid_loader.batch_size + j], labels[j]] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

    plt.imshow(confusion_matrix, cmap='Blues')
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

    plt.show()
    plt.savefig('confusion_matrix.png')