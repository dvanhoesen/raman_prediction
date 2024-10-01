import sys, os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Feed Forward NN
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# Fully Convolutional Neural Network (FCN) except for first layer b/c no connection between input features
class RamanPredictorFCN_fullyConnected1(nn.Module):
    def __init__(self, input_size, output_size, ks=3):
        super(RamanPredictorFCN_fullyConnected1, self).__init__()
        # Assuming input_size is the number of features in the chemical composition
        self.fc1 = nn.Linear(input_size, 64)  # Fully connected layer to project input to 64 features
        #self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=ks, padding=1)
        #self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=ks, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ks, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ks, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=ks, padding=1)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=output_size, kernel_size=ks, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape input for CNN (batch_size, channels, input_size)
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(2)  # Adding a channel dimension (assuming the input is 2D: [batch_size, input_size] becomes [batch_size, input_size, 1])
        #x = self.relu(self.conv1(x)) # [batch_size, 32, 1]
        #x = self.relu(self.conv2(x)) # [batch_size, 64, 1]
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)  # Output layer without ReLU to maintain the full range of the spectrum
        x = x.squeeze(2)  # Remove the channel dimension to return to shape [batch_size, output_size]
        return x


# Fully Convolutional Neural Network (FCN) - upsampling using out channels, i.e., without nn.ConvTranspose1d()
class RamanPredictorFCN(nn.Module):
    def __init__(self, input_size, output_size, ks=3):
        super(RamanPredictorFCN, self).__init__()
        # Assuming input_size is the number of features in the chemical composition
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=ks, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=ks, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ks, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ks, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=ks, padding=1)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=output_size, kernel_size=ks, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape input for CNN (batch_size, channels, input_size)
        x = x.unsqueeze(2)  # Adding a channel dimension (assuming the input is 2D: [batch_size, input_size] becomes [batch_size, input_size, 1])
        x = self.relu(self.conv1(x)) # [batch_size, 32, 1]
        x = self.relu(self.conv2(x)) # [batch_size, 64, 1]
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)  # Output layer without ReLU to maintain the full range of the spectrum
        x = x.squeeze(2)  # Remove the channel dimension to return to shape [batch_size, output_size]
        return x



# Custom DataLoader

class CustomDataset(Dataset):
    def __init__(self, x_inputs_proc, y_labels_proc, add_noise=False):
        """
        Args:
            x_inputs_proc (numpy array): The input data of shape (N, 22).
            y_labels_proc (numpy array): The label data of shape (N, 1024).
            add_noise (bool): Whether to add random noise to the labels.
        """

        self.x_inputs_proc = torch.tensor(x_inputs_proc, dtype=torch.float32)
        self.y_labels_proc = torch.tensor(y_labels_proc, dtype=torch.float32)
        self.add_noise = add_noise

    def __len__(self):
        return len(self.x_inputs_proc)

    def __getitem__(self, idx):
        x = self.x_inputs_proc[idx]
        y = self.y_labels_proc[idx]

        # Add noise to the labels if augmentation is enabled
        if self.add_noise:
            noise_std = torch.FloatTensor(1).uniform_(0, 0.02).item() # random noise magnitude between 0 and 0.02
            noise = torch.randn_like(y) * noise_std
            y = y + noise

        return x, y
