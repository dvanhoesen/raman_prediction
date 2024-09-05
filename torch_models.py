import sys, os
import torch
import torch.nn as nn

# Feed Forward NN
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RamanPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# CNN model with fully connected (FC) layers
class RamanPredictorCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RamanPredictorCNN, self).__init__()
        # Assuming input_size is the number of features in the chemical composition
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_size, 256)
        self.fc2 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        # Reshape input for CNN (batch_size, channels, input_size)
        x = x.unsqueeze(1)  # Adding a channel dimension (assuming the input is 2D: [batch_size, input_size])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Fully Convolutional Neural Network (FCN)
class RamanPredictorFCN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RamanPredictorCNN, self).__init__()
        # Assuming input_size is the number of features in the chemical composition
        # Using a series of Conv1D layers to directly output the Raman spectrum
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=output_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape input for CNN (batch_size, channels, input_size)
        x = x.unsqueeze(1)  # Adding a channel dimension (assuming the input is 2D: [batch_size, input_size])
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)  # Output layer without ReLU to maintain the full range of the spectrum
        x = x.squeeze(1)  # Remove the channel dimension to return to shape [batch_size, output_size]
        return x


