import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Custom Imports
import torch_models as tm 

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
dataset = tm.CustomDataset(x_inputs_proc, y_labels_proc, add_noise=True)

# Create the DataLoader
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example iteration over DataLoader
for x_batch, y_batch in data_loader:
    print(x_batch.shape)  # Shape: (32, 22)
    print(y_batch.shape)  # Shape: (32, 1024)
    break  # Just for demonstration purposes


sys.exit('checking torch imports')



# Create a PyTorch dataset
dataset = TensorDataset(X, y)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Initialize the model, loss function, and optimizer
input_size = chemical_compositions.shape[1]
output_size = raman_spectra.shape[1]

model = tm.FeedForwardNN(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Save the trained model
#torch.save(model.state_dict(), 'raman_predictor.pth')
