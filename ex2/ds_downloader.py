import torch
from torchvision import datasets, transforms
from torch.utils import data as data_utils


def get_dataloaders(batch_size=64, subset_size=100):
    """
    Downloads MNIST and returns train, test, and a small subset train loader.
    """
    transform = transforms.ToTensor()

    # Download and load the datasets
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    # Standard loaders
    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Subset loader (for Q2 and Q3)
    indices = torch.arange(subset_size)
    subset_dataset = data_utils.Subset(train_dataset, indices)
    subset_train_loader = data_utils.DataLoader(subset_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader, subset_train_loader