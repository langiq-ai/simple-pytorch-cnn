import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from model import SimpleCNN

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add the source directory to the system path
sys.path.append("../src")


# Function to train the model
def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    logging.info("Starting training process")
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:  # Log every 100 batches
                logging.debug(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%"
        )

    logging.info("Training process completed")


# Main function to run the training
def main(args):
    logging.info("Starting main training function")
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Transformations for the dataset
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),  # Resize images to 28x28
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize((0.5,), (0.5,)),  # Normalize images
        ]
    )

    # Load dataset (e.g., MNIST)
    if args.dataset == "mnist":
        logging.info("Loading MNIST dataset")
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif args.dataset == "cifar10":
        logging.info("Loading CIFAR-10 dataset")
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        logging.error(f"Unsupported dataset: {args.dataset}")
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    logging.info("DataLoader created")

    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Model, criterion, and optimizer initialized")

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device, args.epochs)

    # Save the model checkpoint
    torch.save(model.state_dict(), "cnn_model.pth")
    logging.info("Model saved as cnn_model.pth")


if __name__ == "__main__":
    # Argument parsing for CLI
    parser = argparse.ArgumentParser(description="Train a CNN model using PyTorch")
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset to use: mnist or cifar10"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )

    args = parser.parse_args()
    logging.info("Arguments parsed successfully")

    main(args)
