
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import matplotlib.pyplot as plt

# Download the dataset
data_flag = 'pneumoniamnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Load the data
train_dataset = DataClass(split='train', download=download)
test_dataset = DataClass(split='test', download=download)

# Get some information about the dataset
print(f"Data Flag: {data_flag}")
print(f"Task: {task}")
print(f"Number of channels: {n_channels}")
print(f"Number of classes: {n_classes}")
print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")
print(f"Labels: {info['label']}")


# Show a few example images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    image, label = train_dataset[i]
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {info['label'][str(label[0])]}")
    plt.axis('off')
plt.show()
