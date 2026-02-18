
import torch
import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_data_loaders(batch_size, data_flag='pneumoniamnist', test_size=0.2, random_state=42):
    """
    Returns train, validation, and test data loaders for the PneumoniaMNIST dataset.

    Args:
        batch_size (int): The batch size for the data loaders.
        data_flag (str): The name of the MedMNIST dataset.
        test_size (float): The proportion of the training set to use for validation.
        random_state (int): The random seed for the train-validation split.

    Returns:
        tuple: A tuple containing the train, validation, and test data loaders.
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Data augmentation and normalization
    data_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Data transformation for the test set (without augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # Load the datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=test_transform, download=True)

    # Create a validation split from the training data
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=test_size,
        random_state=random_state,
        stratify=train_dataset.labels
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Set the transform for the validation subset to the test transform
    val_subset.dataset.transform = test_transform


    # Create the data loaders
    train_loader = DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Example of how to use the function
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")
    
    # You can also inspect a batch
    for images, labels in train_loader:
        print(f"Batch of images has shape: {images.shape}")
        print(f"Batch of labels has shape: {labels.shape}")
        break

