import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# Custom Imports
import torch_models as tm 

sys.exit('checking torch imports')

# chemical_compositions: shape (num_samples, num_features)
# raman_spectra: shape (num_samples, spectrum_length)
chemical_compositions = np.load('chemical_compositions.npy')
raman_spectra = np.load('raman_spectra.npy')

# Convert numpy arrays to PyTorch tensors
X = torch.tensor(chemical_compositions, dtype=torch.float32)
y = torch.tensor(raman_spectra, dtype=torch.float32)

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
torch.save(model.state_dict(), 'raman_predictor.pth')
