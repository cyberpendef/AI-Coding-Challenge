
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import json
import os

from data_loader import get_data_loaders
from model import SimpleCNN

def train(model, train_loader, val_loader, epochs, learning_rate, device, save_path):
    """
    Trains the CNN model.

    Args:
        model (nn.Module): The CNN model to train.
        train_loader (DataLoader): The data loader for the training set.
        val_loader (DataLoader): The data loader for the validation set.
        epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for the optimizer.
        device (torch.device): The device to train the model on (CPU or GPU).
        save_path (str): The path to save the best model and training history.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = torch.sigmoid(outputs) > 0.5
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                predicted = torch.sigmoid(outputs) > 0.5
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct_val / total_val

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")

    # Save the training history
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(history, f)


def main():
    parser = argparse.ArgumentParser(description="Train a CNN for Pneumonia Classification")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save the model and results (default: results)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create save directory if it doesn't exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    train_loader, val_loader, _ = get_data_loaders(args.batch_size)

    model = SimpleCNN().to(device)

    train(model, train_loader, val_loader, args.epochs, args.lr, device, args.save_path)

if __name__ == '__main__':
    main()
