import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import random

# Custom Imports
import torch_models as tm 
import torch_functions as tf


# Seeding for Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# CUDA seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)  # For multiple GPUs


# Check if CUDA is available (MPS for macos with M1 chip)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print("Device: ", device)

# Load numpy files
basepath = "train_data" + os.path.sep

y_labels_proc = np.load(basepath + "y_labels_proc.npy", allow_pickle=True)
x_inputs_proc = np.load(basepath + "x_inputs_proc.npy", allow_pickle=True)
names_proc_all = np.load(basepath + "names_proc.npy", allow_pickle=True)
rruffid_proc_all = np.load(basepath + "rruffid_proc.npy", allow_pickle=True)

print("\nFinal Processed Data shapes")
print("y_labels_proc shape: ", y_labels_proc.shape)
print("x_inputs_proc shape: ", x_inputs_proc.shape)
print("names_proc_all shape: ", names_proc_all.shape)
print("rruffid_proc_all shape: ", rruffid_proc_all.shape)


# Create the dataset with data augmentation (adding noise)
full_dataset = tm.CustomDataset(x_inputs_proc, y_labels_proc, add_noise=True)

# Split dataset sizes (e.g., 70% train, 15% validation, 15% test)
train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create DataLoaders for each set
batch_size = [16, 32, 64]
learning_rate = [0.003, 0.001, 0.0003]
optimizer = ['Adam', 'NAdam', 'RMSprop', 'SGD']


parameters = []
count = 0
for bs in batch_size: 
    for lr in learning_rate:
        for opt in optimizer:

            parameters.append({
                "batch_size": bs,
                "learning_rate": lr,
                "optimizer": opt
            })

            count += 1


input_size = x_inputs_proc.shape[1]
output_size = 1024

epochs = 100
criterion = nn.MSELoss()
ks = 3

parameters_with_losses = []
count = 0
for params in parameters:

    lr = params['learning_rate']
    batch_size = params['batch_size']
    opt = params['optimizer']

    print("{} / {}".format(count+1, len(parameters)), lr, batch_size, opt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model = tm.RamanPredictorFCConvTranspose1d(input_size, output_size, ks=ks)
    #model = tm.RamanPredictorFCN(input_size, output_size, ks=ks)
    #model = tm.FeedForwardNN(input_size, output_size)
    model = tm.FeedForwardNN_CNN(input_size, output_size, ks=ks)

    # Send model to OS cuda device (M1 Mac OS is mps)
    model.to(device)

    if opt == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    if opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    if opt == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=lr)
    if opt == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    

    # Train the model
    train_losses, val_losses = tf.train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    last_train_loss = train_losses[-1]
    last_val_loss = val_losses[-1]

    params['train_loss'] = last_train_loss
    params['val_loss'] = last_val_loss

    parameters_with_losses.append(params)

    print(params)

    count += 1



sorted_data = sorted(parameters_with_losses, key=lambda x: x['val_loss'], reverse=False)

print("\nVal Loss\tTrain Loss\tLR\tOptimizer\tbatch size")
for row in sorted_data:
    val_loss = row['val_loss']
    train_loss = row['train_loss']
    lr = row['learning_rate']
    batch_size = row['batch_size']
    opt = row['optimizer']
    
    print("{:.6f}\t{:.6f}\t{}\t{}\t\t{}".format(val_loss, train_loss, lr, opt, batch_size))



