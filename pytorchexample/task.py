"""pytorchexample: A Flower / PyTorch app."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, RandomCrop, RandomHorizontalFlip
from torchvision.models import resnet18, ResNet18_Weights

from medmnist.dataset import BloodMNIST

DATA_DIR = Path("./data/.medmnist")  # Directory to store the MedMNIST dataset
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)

    def forward(self, x):
        return self.model(x)    


fds = None  # Cache FederatedDataset

# pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_transforms = Compose([
    Resize((32, 32)),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
])

test_transforms = Compose([
    Resize((32, 32)),
    ToTensor(),
    Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
])

train_dataset = BloodMNIST(
    split="train",
    transform=train_transforms,
    download=False,
    root=DATA_DIR,
)

test_dataset = BloodMNIST(
    split="test",
    transform=test_transforms,
    download=False,
    root=DATA_DIR,
)

def apply_transforms(batch, transform):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
):
    """Load one IID partition of BloodMNIST."""

    indices = np.arange(len(train_dataset))

    # Same partitioning every time
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    # Divide indices among clients
    partitions = np.array_split(
        indices,
        num_partitions,
    )

    client_indices = partitions[partition_id]

    client_dataset = Subset(
        train_dataset,
        client_indices,
    )

    trainloader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Client-local validation set
    val_size = int(0.2 * len(client_dataset))

    train_size = len(client_dataset) - val_size

    client_train, client_val = torch.utils.data.random_split(
        client_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(
        client_train,
        batch_size=batch_size,
        shuffle=True,
    )

    testloader = DataLoader(
        client_val,
        batch_size=batch_size,
        shuffle=False,
    )

    return trainloader, testloader

def load_centralized_dataset():
    """Load the complete BloodMNIST test set."""

    return DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
    )


def train(net, trainloader, epochs, lr, device):
    net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0.9,
    )

    net.train()

    running_loss = 0.0

    for _ in range(epochs):
        for images, labels in trainloader:

            images = images.to(device)

            # MedMNIST labels can have shape [batch, 1]
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()

            outputs = net(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / (
        epochs * len(trainloader)
    )


def test(net, testloader, device):
    net.to(device)

    criterion = nn.CrossEntropyLoss()

    net.eval()

    correct = 0
    total_loss = 0.0

    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = net(images)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

    accuracy = correct / len(testloader.dataset)

    loss = total_loss / len(testloader)

    return loss, accuracy
