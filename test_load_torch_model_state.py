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

# CUDA device
device = torch.device("mps")
print("Device: ", device)

# Load numpy files
basepath_data = "train_data_wavenumber_cutoffs" + os.path.sep
basepath_model = "results_trained_FeedForward_batch_size_1" + os.path.sep
savename_model = basepath_model + "FeedForwardNN_test_weights.pth"
savename_optimizer = basepath_model + "FeedForwardNN_test_optimizer.pth"

y_labels_proc = np.load(basepath_data + "y_labels_proc.npy", allow_pickle=True)
x_inputs_proc = np.load(basepath_data + "x_inputs_proc.npy", allow_pickle=True)
names_proc_all = np.load(basepath_data + "names_proc.npy", allow_pickle=True)
rruffid_proc_all = np.load(basepath_data + "rruffid_proc.npy", allow_pickle=True)

print("\nFinal Processed Data shapes")
print("y_labels_proc shape: ", y_labels_proc.shape)
print("x_inputs_proc shape: ", x_inputs_proc.shape)
print("names_proc_all shape: ", names_proc_all.shape)
print("rruffid_proc_all shape: ", rruffid_proc_all.shape)


# Extract indexes of input and labels used for training the model
idxs_train = np.load(basepath_model + "idxs_train.npy")
idxs_val = np.load(basepath_model + "idxs_val.npy")
idxs_test = np.load(basepath_model + "idxs_test.npy")

## Example extract inputs and labels for test data
y_labels_proc_test = y_labels_proc[idxs_test]
x_inputs_proc_test = x_inputs_proc[idxs_test]
names_proc_all_test = names_proc_all[idxs_test]
rruffid_proc_all_test = rruffid_proc_all[idxs_test]

print("\nTest Data shapes")
print("y_labels_proc shape: ", y_labels_proc_test.shape)
print("x_inputs_proc shape: ", x_inputs_proc_test.shape)
print("names_proc_all shape: ", names_proc_all_test.shape)
print("rruffid_proc_all shape: ", rruffid_proc_all_test.shape)


## Example load and run the model (** ALL PARAMETERS  **)

# Parameters ( ** MUST MATCH THOSE USED MODEL TRAINING ** )
batch_size = 1
input_size = x_inputs_proc.shape[1]
output_size = 1024
model = tm.FeedForwardNN(input_size, output_size)
model.to(device)
optimizer = optim.NAdam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Mean Squared Error Loss

# Load model
model, optimizer = tf.load_model(model, optimizer, savename_model, savename_optimizer, device)

# Build the test loader using the custom dataset loader without noise added on the test data
test_dataset = tm.CustomDataset(x_inputs_proc_test, y_labels_proc_test, add_noise=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set
x_test, y_pred, y_true = tf.evaluate_model(model, test_loader, criterion, device)

# Plot random spec results
x_test = x_test.cpu()
y_pred = y_pred.cpu()
y_true = y_true.cpu()

tf.plot_random_predictions(x_test, y_pred, y_true, num_samples=3)

