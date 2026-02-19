
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import PneumoniaMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import seaborn as sns
import os

# Define paths
REPORTS_DIR = 'reports'
MODELS_DIR = 'models'
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Data augmentation and normalization
data_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Load the dataset
train_dataset = PneumoniaMNIST(split='train', transform=data_transform, download=True)
val_dataset = PneumoniaMNIST(split='val', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]), download=True)
test_dataset = PneumoniaMNIST(split='test', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]), download=True)

# Create data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, train_loader, val_loader, epochs=25):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for images, labels in train_loader:
            labels = labels.squeeze().long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.squeeze().long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct / total)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.2f}%')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.squeeze().long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            y_scores.extend(outputs.softmax(dim=1)[:, 1].tolist())
    
    return y_true, y_pred, y_scores

# Plotting functions
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(REPORTS_DIR, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    plt.close()

# Main execution
if __name__ == '__main__':
    model = SimpleCNN()
    
    # Train the model
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_loader, val_loader)
    
    # Save the model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'pneumonia_cnn.pth'))
    
    # Evaluate the model
    y_true, y_pred, y_scores = evaluate(model, test_loader)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print(f'AUC: {roc_auc:.4f}')
    
    # Plot results
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curve(y_true, y_scores)
    
    # Failure case analysis
    misclassified_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    
    # Create markdown report
    report = f"""
# Task 1: CNN Classification Report

## Model Architecture
A simple CNN was used with two convolutional layers, each followed by max pooling, and two fully connected layers.

## Training Methodology
- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 25
- **Batch Size:** 64
- **Data Augmentation:** Random rotation (10 degrees), random translation (10%).

## Evaluation Metrics
- **Accuracy:** {accuracy:.4f}
- **Precision:** {precision:.4f}
- **Recall:** {recall:.4f}
- **F1-score:** {f1:.4f}
- **AUC:** {roc_auc:.4f}

## Visualizations
### Training Curves
![Training Curves](training_curves.png)

### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

### ROC Curve
![ROC Curve](roc_curve.png)

## Failure Case Analysis
The model misclassified {len(misclassified_indices)} out of {len(y_true)} test images. 
(Further analysis would involve visualizing these images and investigating patterns.)
"""
    with open(os.path.join(REPORTS_DIR, 'task1_classification_report.md'), 'w') as f:
        f.write(report)
        
    print("Task 1 finished. Report and plots are saved in the 'reports' directory.")
