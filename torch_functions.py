import torch
import random
import matplotlib.pyplot as plt

# Training and validation loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=20):
    train_losses = [] 
    val_losses = []

    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Training loop
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(x_batch)  # Forward pass
            loss = criterion(outputs, y_batch)  # Compute loss
            loss.backward()  # Backward pass (calculate gradients)
            optimizer.step()  # Update weights
            running_loss += loss.item() * x_batch.size(0)  # Accumulate loss

        # Average training loss
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for validation
            for x_batch, y_batch in val_loader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(x_batch)  # Forward pass
                loss = criterion(outputs, y_batch)  # Compute loss
                val_loss += loss.item() * x_batch.size(0)

        # Average validation loss
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Print epoch results
        print(f"Epoch [{epoch+1}/{epochs}] - Training Loss: {epoch_train_loss:.5f} - Validation Loss: {epoch_val_loss:.5f}")
    
    return train_losses, val_losses



# Test the model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    all_inputs = []
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
        
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0)

            all_inputs.append(x_batch)
            all_outputs.append(outputs)
            all_labels.append(y_batch)

    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    return torch.cat(all_inputs), torch.cat(all_outputs), torch.cat(all_labels)



# Function to plot predicted vs known values
def plot_random_predictions(x_test, y_pred, y_true, num_samples=4):
    
    colors = ['black', 'blue', 'green', 'red', 'orange', 'purple']
    indices = random.sample(range(len(x_test)), num_samples)


    
    for i, idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=(11,5))
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel('Normalized Intensity', fontsize=12)

        #color = colors[i]

        # Plot known values
        ax.plot(y_true[idx].numpy(), label='{} - True'.format(idx))
        
        # Plot predicted values
        ax.plot(y_pred[idx].numpy(), label='{} - Predicted'.format(idx), linestyle='--')
        
        plt.legend(loc='best')
        plt.show()


# Function to plot train and val losses as a function of epoch
def plot_losses(train_losses, val_losses):

    fig, ax = plt.subplots(figsize=(11,5))
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Criterion (MSE)', fontsize=12)
    ax.plot(train_losses, label='Train', color='blue')
    ax.plot(val_losses, label='Validation', color='red')
    plt.legend(loc='best')
    plt.show()

