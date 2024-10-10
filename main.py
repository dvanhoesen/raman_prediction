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
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Instantiate the model, loss function, and optimizer
input_size = x_inputs_proc.shape[1]
output_size = 1024

model = tm.RamanPredictorFCN(input_size, output_size)
#model = tm.RamanPredictorFCN_fullyConnected1(input_size, output_size)
#model = tm.FeedForwardNN(input_size, output_size)
#model = tm.RamanPredictorFCConvTranspose1d(input_size, output_size, ks=3)

# Send model to OS cuda device (M1 Mac OS is mps)
model.to(device)

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.003)
criterion = nn.MSELoss()  # Mean Squared Error Loss
epochs = 150

"""
optimizers = {
    "Adadelta": optim.Adadelta(model.parameters(), lr=0.001),
    "Adagrad": optim.Adagrad(model.parameters(), lr=0.001),
    "Adam": optim.Adam(model.parameters(), lr=0.001),
    "AdamW": optim.AdamW(model.parameters(), lr=0.001),
    "Adamax": optim.Adamax(model.parameters(), lr=0.001),
    "ASGD": optim.ASGD(model.parameters(), lr=0.001),
    "NAdam": optim.NAdam(model.parameters(), lr=0.001),
    "RAdam": optim.RAdam(model.parameters(), lr=0.001),
    "RMSprop": optim.RMSprop(model.parameters(), lr=0.001),
    "Rprop": optim.Rprop(model.parameters(), lr=0.001),
    "SGD": optim.SGD(model.parameters(), lr=0.001),
}


all_train_losses = {}
all_val_losses = {}

# Loop through all of the optmizers
for name, optimizer in optimizers.items():
    print(f"Training with {name} optimizer...")
    model = tm.RamanPredictorFCConvTranspose1d(input_size, output_size, ks=3)

    optimizers_new = {
        "Adadelta": optim.Adadelta(model.parameters(), lr=0.001),
        "Adagrad": optim.Adagrad(model.parameters(), lr=0.001),
        "Adam": optim.Adam(model.parameters(), lr=0.001),
        "AdamW": optim.AdamW(model.parameters(), lr=0.001),
        "Adamax": optim.Adamax(model.parameters(), lr=0.001),
        "ASGD": optim.ASGD(model.parameters(), lr=0.001),
        "NAdam": optim.NAdam(model.parameters(), lr=0.001),
        "RAdam": optim.RAdam(model.parameters(), lr=0.001),
        "RMSprop": optim.RMSprop(model.parameters(), lr=0.001),
        "Rprop": optim.Rprop(model.parameters(), lr=0.001),
        "SGD": optim.SGD(model.parameters(), lr=0.001),
    }

    optimizer = optimizers_new[name]
    
    train_losses, val_losses = tf.train_model(model, train_loader, val_loader, criterion, optimizer, epochs)

    all_train_losses[name] = train_losses
    all_val_losses[name] = val_losses


fig, ax = plt.subplots(figsize=(11,5))
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Criterion (MSE)', fontsize=12)
ax.set_title("Train Loss")
for name, losses in all_train_losses.items():
    ax.plot(losses, label=name)
ax.legend(loc='best')


fig2, ax2 = plt.subplots(figsize=(11,5))
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Criterion (MSE)', fontsize=12)
ax2.set_title("Validation Loss")
for name, losses in all_val_losses.items():
    ax2.plot(losses, label=name)
ax2.legend(loc='best')
"""



# Train the model
train_losses, val_losses = tf.train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

# Evaluate the model on the test set
x_test, y_pred, y_true = tf.evaluate_model(model, test_loader, criterion, device)

# Plot random spec results
x_test = x_test.cpu()
y_pred = y_pred.cpu()
y_true = y_true.cpu()

tf.plot_random_predictions(x_test, y_pred, y_true, num_samples=10)

# plot train and val losses with epoch
fig_loss, ax_loss = tf.plot_losses(train_losses, val_losses)

print("can see structure appearing, but the validation loss stagnates around 150 epochs")
print("possible that the method using the lower and upper bound wavelengths in the input is not sufficient")
print("may need to revert to interpolating between a range and adding zeros where necessary as padding. ")

plt.show()


