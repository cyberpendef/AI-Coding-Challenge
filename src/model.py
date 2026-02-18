
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    A simple CNN model for binary classification.
    """
    def __init__(self, in_channels=1, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # Example of how to use the model
    model = SimpleCNN()
    dummy_input = torch.randn(64, 1, 28, 28) # (batch_size, channels, height, width)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Check if the output shape is as expected
    assert output.shape == (64, 1), f"Expected output shape (64, 1), but got {output.shape}"
