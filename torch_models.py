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
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc5 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x


# Feed Forward NN with CNNS
class FeedForwardNN_CNN(nn.Module):
    def __init__(self, input_size, output_size, ks=3):
        super(FeedForwardNN_CNN, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, output_size)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=ks, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=ks, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=ks, padding=1)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=ks, padding=1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):

        # Fully connected layers to bring to the correct shape [batch_size, input_size]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        # Reshape the input to add a channel dimension for Conv1d [batch_size, 1, input_size]
        x = x.unsqueeze(1)

        # Apply the convolution layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        x = x.squeeze(1)  # Remove the channel dimension to return to shape [batch_size, output_size]
        
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
        #self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=ks, padding=0)
        #self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=ks, padding=0)
        #self.conv3 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=ks, padding=0)
        self.fc1 = nn.Linear(input_size, 32)  # Fully connected layer to project input to 32 features
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=ks, padding=0)
        self.conv5 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=ks, padding=0)
        self.conv6 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=ks, padding=0)
        self.conv7 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=ks, padding=0)
        self.conv8 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=ks, padding=0)
        self.conv9 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=ks, padding=0)
        self.conv10 = nn.Conv1d(in_channels=512, out_channels=output_size, kernel_size=ks, padding=0)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape input for CNN (batch_size, channels, input_size)
        x = self.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Adding a channel dimension (assuming the input is 2D: [batch_size, input_size] becomes [batch_size, input_size, 1])
        #x = self.relu(self.conv1(x)) # [batch_size, 32, 1]
        #x = self.relu(self.conv2(x)) # [batch_size, 64, 1]
        #x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))

        x = self.conv10(x)  # Output layer without ReLU to maintain the full range of the spectrum
        x = torch.mean(x, -1) # Average last column
        return x


# Fully Convolutional Neural Network (FCN) - upsampling with nn.ConvTranspose1d()
class RamanPredictorFCConvTranspose1d(nn.Module):
    def __init__(self, input_size, output_size, ks=3):
        super(RamanPredictorFCConvTranspose1d, self).__init__()

        # Define fully connected layers to process initial input features
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)

        # Define ConvTranspose1d layers to upsample the sequence length
        self.deconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=64, kernel_size=ks, stride=1, padding=1) # 128
        self.deconv2 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=ks, stride=2, padding=1) # 256
        self.deconv3 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=ks, stride=2, padding=0) # 512
        self.deconv4 = nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=ks, stride=2, padding=0, output_padding=1) # 1024
        self.deconv5 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=ks, stride=1, padding=1)  # 1024

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        # Fully connected layers to process input of shape [input_size, 22]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x))

        x = x.unsqueeze(1)

        # ConvTranspose1d layers to upsample the sequence length
        x = self.relu(self.deconv1(x))  # [input_size, 32, 2]
        x = self.relu(self.deconv2(x))  # [input_size, 32, 4]
        x = self.relu(self.deconv3(x))  # [input_size, 32, 8]
        x = self.relu(self.deconv4(x))  # [input_size, 32, 16]
        x = self.relu(self.deconv5(x))  # [input_size, 32, 1024] (final upsampling)

        # Output will be of shape [input_size, 1024]
        x = x.squeeze(1)

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
            noise_std = torch.FloatTensor(1).uniform_(0, 0.01).item() # random noise magnitude between 0 and 0.01
            noise = torch.randn_like(y) * noise_std
            y = y + noise

        return x, y
