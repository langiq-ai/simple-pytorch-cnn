import torch
import torch.nn as nn
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to evaluate the model
def evaluate_model(model, test_loader, device):
    logging.info("Starting model evaluation")
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            logging.debug(f"Processed batch: Total={total}, Correct={correct}")

    accuracy = 100 * correct / total
    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
    print(f"Accuracy: {accuracy:.2f}%")


# Main function for evaluation
def main(args):
    logging.info("Starting main evaluation function")
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
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif args.dataset == "cifar10":
        logging.info("Loading CIFAR-10 dataset")
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        logging.error(f"Unsupported dataset: {args.dataset}")
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False
    )
    logging.info("DataLoader created")

    # Initialize the model and load the trained weights
    model = SimpleCNN().to(device)
    logging.info(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logging.info("Model loaded successfully")

    # Evaluate the model
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    # Argument parsing for CLI
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN model using PyTorch"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset to evaluate: mnist or cifar10",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (e.g., cnn_model.pth)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for evaluation"
    )

    args = parser.parse_args()
    logging.info("Arguments parsed successfully")

    main(args)
