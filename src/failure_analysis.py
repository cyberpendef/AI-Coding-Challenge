
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from data_loader import get_data_loaders
from model import SimpleCNN
from medmnist import INFO

def failure_analysis(model, test_loader, device, save_path):
    """
    Performs failure analysis by saving misclassified images.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): The data loader for the test set.
        device (torch.device): The device to run the model on.
        save_path (str): The path to save the misclassified images.
    """
    model.eval()
    false_positives = []
    false_negatives = []
    
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    class_names = info['label']

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            # Find misclassified images
            misclassified_indices = (preds != labels).squeeze()
            
            for j in range(len(misclassified_indices)):
                if misclassified_indices[j]:
                    image = images[j].cpu().numpy().squeeze()
                    true_label = int(labels[j].item())
                    pred_label = int(preds[j].item())
                    
                    if pred_label == 1 and true_label == 0: # False Positive
                        if len(false_positives) < 5:
                            false_positives.append((image, true_label, pred_label))
                    
                    elif pred_label == 0 and true_label == 1: # False Negative
                        if len(false_negatives) < 5:
                            false_negatives.append((image, true_label, pred_label))
            
            if len(false_positives) >= 5 and len(false_negatives) >= 5:
                break

    # Create directory for failure cases
    failure_cases_path = os.path.join(save_path, 'failure_cases')
    if not os.path.exists(failure_cases_path):
        os.makedirs(failure_cases_path)

    # Save false positives
    for i, (image, true_label, pred_label) in enumerate(false_positives):
        plt.imshow(image, cmap='gray')
        plt.title(f'FP {i+1}: True: {class_names[str(true_label)]}, Pred: {class_names[str(pred_label)]}')
        plt.savefig(os.path.join(failure_cases_path, f'false_positive_{i+1}.png'))
        plt.close()

    # Save false negatives
    for i, (image, true_label, pred_label) in enumerate(false_negatives):
        plt.imshow(image, cmap='gray')
        plt.title(f'FN {i+1}: True: {class_names[str(true_label)]}, Pred: {class_names[str(pred_label)]}')
        plt.savefig(os.path.join(failure_cases_path, f'false_negative_{i+1}.png'))
        plt.close()

    print(f"Saved 5 false positives and 5 false negatives to {failure_cases_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform failure analysis on a CNN for Pneumonia Classification")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for evaluation (default: 64)')
    parser.add_argument('--model_path', type=str, default='results/best_model.pth', help='Path to the saved model (default: results/best_model.pth)')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save the failure cases (default: results)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_data_loaders(args.batch_size)
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    failure_analysis(model, test_loader, device, args.save_path)

if __name__ == '__main__':
    main()
