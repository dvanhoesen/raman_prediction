import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# Custom Imports
import torch_models as tm 
import torch_functions as tf

# Load numpy files
basepath = "train_data" + os.path.sep

"""
y_labels_raw = np.load(basepath + "y_labels_raw.npy", allow_pickle=True)
x_inputs_raw = np.load(basepath + "x_inputs_raw.npy", allow_pickle=True)
names_raw_all = np.load(basepath + "names_raw.npy", allow_pickle=True)
rruffid_raw_all = np.load(basepath + "rruffid_raw.npy", allow_pickle=True)

print("\nRaw Data shapes")
print("y_labels_raw shape: ", y_labels_raw.shape)
print("x_inputs_raw shape: ", x_inputs_raw.shape)
print("names_raw_all shape: ", names_raw_all.shape)
print("rruffid_raw_all shape: ", rruffid_raw_all.shape)
"""

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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Instantiate the model, loss function, and optimizer
input_size = x_inputs_proc.shape[1]
output_size = 1024

#model = tm.RamanPredictorFCN(input_size, output_size)
#model = tm.RamanPredictorFCN_fullyConnected1(input_size, output_size)
model = tm.FeedForwardNN(input_size, output_size)

criterion = nn.MSELoss()  # Mean Squared Error Loss

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.NAdam(model.parameters(), lr=0.001)

epochs = 200

# Train the model
train_losses, val_losses = tf.train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs)

# Evaluate the model on the test set
x_test, y_pred, y_true = tf.evaluate_model(model, test_loader, criterion)



print("Training Works, but should try both FCN with and without nn.ConvTranspose1d(), which is meant for upsampling")
print("Data Leakage b/c many of the same type of mineral in train, test, and validation sets")

# Plot random spec results
fig_pred, ax_pred = tf.plot_random_predictions(x_test, y_pred, y_true, num_samples=3)

# plot train and val losses with epoch
fig_loss, ax_loss = tf.plot_losses(train_losses, val_losses)


plt.show()


