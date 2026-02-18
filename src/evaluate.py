
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
import argparse
import json
import os
import seaborn as sns

from data_loader import get_data_loaders
from model import SimpleCNN

def plot_confusion_matrix(cm, classes, save_path):
    """
    Plots the confusion matrix.
    """
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """
    Plots the ROC curve.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))
    plt.close()

def plot_training_history(history, save_path):
    """
    Plots the training and validation loss and accuracy.
    """
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(save_path, 'training_curves.png'))
    plt.close()

def evaluate(model, test_loader, device, save_path):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): The data loader for the test set.
        device (torch.device): The device to run the model on.
        save_path (str): The path to save the evaluation results.
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=['normal', 'pneumonia'], save_path=save_path)

    # ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, save_path)
    
    # Training history
    with open(os.path.join(save_path, 'history.json'), 'r') as f:
        history = json.load(f)
    plot_training_history(history, save_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a CNN for Pneumonia Classification")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for evaluation (default: 64)')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth', help='Path to the saved model (default: results/best_model.pth)')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save the evaluation results (default: results)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_data_loaders(args.batch_size)
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    evaluate(model, test_loader, device, args.save_path)

if __name__ == '__main__':
    main()
