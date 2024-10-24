import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # First convolutional layer: input channels = 1, output channels = 16, kernel size = 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        logging.debug(
            "Initialized first convolutional layer with 1 input channel, 16 output channels, and kernel size 3"
        )

        # Second convolutional layer: input channels = 16, output channels = 32, kernel size = 3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        logging.debug(
            "Initialized second convolutional layer with 16 input channels, 32 output channels, and kernel size 3"
        )

        # Fully connected layer: output 10 classes (e.g., for classification like MNIST)
        self.fc1 = nn.Linear(
            in_features=32 * 6 * 6, out_features=128
        )  # Assuming 28x28 input image
        logging.debug(
            "Initialized first fully connected layer with input features 32*6*6 and output features 128"
        )

        self.fc2 = nn.Linear(in_features=128, out_features=10)
        logging.debug(
            "Initialized second fully connected layer with input features 128 and output features 10"
        )

    def forward(self, x):
        logging.debug(f"Input shape: {x.shape}")

        # Convolution -> ReLU -> Max pooling
        x = F.relu(self.conv1(x))
        logging.debug(f"After conv1 and ReLU: {x.shape}")

        x = F.max_pool2d(x, kernel_size=2)
        logging.debug(f"After max pooling: {x.shape}")

        # Second convolution -> ReLU -> Max pooling
        x = F.relu(self.conv2(x))
        logging.debug(f"After conv2 and ReLU: {x.shape}")

        x = F.max_pool2d(x, kernel_size=2)
        logging.debug(f"After second max pooling: {x.shape}")

        # Flatten the tensor
        x = x.view(-1, 32 * 6 * 6)
        logging.debug(f"After flattening: {x.shape}")

        # Fully connected layer -> ReLU
        x = F.relu(self.fc1(x))
        logging.debug(f"After fc1 and ReLU: {x.shape}")

        # Output layer (no activation since it's typically used with CrossEntropyLoss)
        x = self.fc2(x)
        logging.debug(f"After fc2 (output layer): {x.shape}")

        return x


# Example of creating the model and moving it to GPU if available
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.debug(f"Model moved to device: {device}")

print(model)
