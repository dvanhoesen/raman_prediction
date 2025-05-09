import sys, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Custom Imports
import torch_models as tm 
import torch_functions as tf


# Check if MPS (Mac Metal Performance Shaders) is available MacOS
if torch.backends.mps.is_available():
    device = torch.device("mps")

# Check if CUDA (NVIDIA GPU) is available
elif torch.cuda.is_available():
    device = torch.device("cuda")

# Default to CPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


include_density_hardness = False
print("Include density and hardness: ", include_density_hardness)


# Load numpy files
if include_density_hardness:
    basepath = "train_data_wavenumber_cutoffs_density_hardness" + os.path.sep
    savepath = "results_trained_FeedForward_density_hardness_unique_names_normed" + os.path.sep
    #savepath = "results_trained_fullyConnected1_density_hardness_unique_names" + os.path.sep
    #savepath = "results_trained_FCN_density_hardness_unique_names" + os.path.sep

else: 
    basepath = "train_data_wavenumber_cutoffs_density_hardness" + os.path.sep # same path as with density and hardness, but will simply leave off the last two features
    savepath = "results_trained_FeedForward_unique_names_normed" + os.path.sep


print("BASEPATH: ", basepath)
print("SAVEPATH: ", savepath)

y_labels_proc = np.load(basepath + "y_labels_proc.npy", allow_pickle=True)
x_inputs_proc = np.load(basepath + "x_inputs_proc.npy", allow_pickle=True)
names_proc_all = np.load(basepath + "names_proc.npy", allow_pickle=True)
rruffid_proc_all = np.load(basepath + "rruffid_proc.npy", allow_pickle=True)

# remove density and hardness if training without it
if not include_density_hardness:
    x_inputs_proc = x_inputs_proc[:, 0:20]

print("\nFinal Processed Data shapes")
print("y_labels_proc shape: ", y_labels_proc.shape)
print("x_inputs_proc shape: ", x_inputs_proc.shape)
print("names_proc_all shape: ", names_proc_all.shape)
print("rruffid_proc_all shape: ", rruffid_proc_all.shape)
print("unique rruffid_proc_all shape: ", np.unique(rruffid_proc_all).shape)

num_samples = x_inputs_proc.shape[0]
un = np.unique(names_proc_all)
uids = np.unique(rruffid_proc_all)

print("Number of samples: ", num_samples)
print("Number of unique names: ", len(un))
print("number of unique ids: ", len(uids))


means_prior = np.mean(x_inputs_proc, axis=0)
stddev_proir = np.std(x_inputs_proc, axis=0)

print("\naverages prior to scaling (last 4): ", means_prior[-4:])
print("stddev prior to scaling (last 4): ", stddev_proir[-4:])

# Standardize the input
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_inputs_proc)

means_after = np.mean(x_scaled, axis=0)
stddev_after = np.std(x_scaled, axis=0)

print("averages after scaling (last 4): ", means_after[-4:])
print("stddev after scaling (last 4): ", stddev_after[-4:])


sys.exit()

batch_size = 256
input_size = x_scaled.shape[1]
output_size = 1024
kernel_size = 3
criterion = nn.MSELoss()
epochs = 500

count_un = 0

for i in range(len(un)):

    left_out_name = un[i]

    # Only training for Quartz as a checker
    #if not left_out_name.startswith("Q"):
    #    continue
    
    idx = np.where(names_proc_all != left_out_name)
    idx_test = np.where(names_proc_all == left_out_name)

    x = x_scaled[idx]
    y = y_labels_proc[idx]

    x_test = x_scaled[idx_test]
    y_test = y_labels_proc[idx_test]

    names_test = names_proc_all[idx_test]
    rruffid_test = rruffid_proc_all[idx_test]

    num_left_out_samples = x.shape[0]
    print("\n{} - {}".format(left_out_name, num_samples - num_left_out_samples))

    # Create the dataset with data augmentation (adding noise)
    full_dataset = tm.CustomDataset(x, y, add_noise=True)
    test_dataset = tm.CustomDataset(x_test, y_test, add_noise=False)

    data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #model = tm.RamanPredictorFCN(input_size, output_size)
    #model = tm.RamanPredictorFCN_fullyConnected1(input_size, output_size)
    model = tm.FeedForwardNN(input_size, output_size)
    #model = tm.RamanPredictorFCConvTranspose1d(input_size, output_size, ks=kernel_size)
    #model = tm.FeedForwardNN_CNN(input_size, output_size, ks=kernel_size)

    # model.apply(tm.init_weights) # Adding initial weights made the model worse

    # Send model to OS cuda device (M1 Mac OS is mps)
    model.to(device)

    optimizer = optim.NAdam(model.parameters(), lr=0.002)
    train_losses = tf.train_model_train_only(model, data_loader, criterion, optimizer, device, epochs=epochs)

    # Evaluate the model on the test set
    y_pred = tf.evaluate_model_single_mineral(model, test_loader, device)
    y_pred = y_pred.cpu().numpy()

    if count_un==0:
        all_predictions = y_pred
        all_spec = y_test
        all_chem = x_test
        all_names = names_test
        all_rruffid = rruffid_test

    else:
        all_predictions = np.append(all_predictions, y_pred, axis=0)
        all_spec = np.append(all_spec, y_test, axis=0)
        all_chem = np.append(all_chem, x_test, axis=0)
        all_names = np.append(all_names, names_test)
        all_rruffid = np.append(all_rruffid, rruffid_test)


    mse = np.mean((y_test - y_pred) ** 2, axis=1, keepdims=True)  # Shape: (N, 1)
    #rmse = np.sqrt(mse)  # Shape: (N, 1)

    print("MSE: ", mse)

    #if i==2:
    #    break

    count_un += 1



print("final all_pred shape: ", all_predictions.shape)
print("final all_spec shape: ", all_spec.shape)
print("final all_chem shape: ", all_chem.shape)
print("final all_names shape: ", all_names.shape)
print("final ids shape: ", all_rruffid.shape)

np.save(savepath + "all_spectra_predictions.npy", all_predictions, allow_pickle=True)
np.save(savepath + "all_spectra.npy", all_spec, allow_pickle=True)
np.save(savepath + "all_chem.npy", all_chem, allow_pickle=True)
np.save(savepath + "all_names.npy", all_names, allow_pickle=True)
np.save(savepath + "all_rruffid.npy", all_rruffid, allow_pickle=True)

