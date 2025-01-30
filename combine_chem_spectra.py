import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
import random


#basepath = 'extracted_data' + os.path.sep
basepath = 'extracted_data_wavenumber_cutoffs' + os.path.sep

#savepath = "train_data" + os.path.sep
savepath = "train_data_wavenumber_cutoffs_density_hardness" + os.path.sep

#xplot = np.arange(1024)
xplot = np.linspace(50, 1700, 1024, endpoint=True)

#flag_include_min_max = True
flag_include_min_max = False

# Load spectra arrays
names_raw = np.load(basepath + "names_raw.npy", allow_pickle=True)
rruffids_raw = np.load(basepath + "rruffids_raw.npy", allow_pickle=True)
x_components_raw = np.load(basepath + "x_components_raw.npy", allow_pickle=True)
y_components_raw = np.load(basepath + "y_components_raw.npy", allow_pickle=True)

names_proc = np.load(basepath + "names_proc.npy", allow_pickle=True)
rruffids_proc = np.load(basepath + "rruffids_proc.npy", allow_pickle=True)
x_components_proc = np.load(basepath + "x_components_proc.npy", allow_pickle=True)
y_components_proc = np.load(basepath + "y_components_proc.npy", allow_pickle=True)

print("\nRaw Files")
print("names_raw shape:", names_raw.shape)
print("rruffids_raw shape:", rruffids_raw.shape)
print("x_components_raw shape:", x_components_raw.shape)
print("y_components_raw shape:", y_components_raw.shape)

print("\nProcessed Files")
print("names_proc shape:", names_proc.shape)
print("rruffids_proc shape:", rruffids_proc.shape)
print("x_components_proc shape:", x_components_proc.shape)
print("y_components_proc shape:", y_components_proc.shape)


# Load chem arrays
chem = np.load("extracted_chemistry_avg.npy", allow_pickle=True)
rruffids_chem = np.load("extracted_chemistry_spec_ids.npy", allow_pickle=True)
names_chem = np.load("extracted_chemistry_spec_names.npy", allow_pickle=True)

# Load density and hardness arrays
names_density_hardness = np.load("extracted_names_density_hardness.npy", allow_pickle=True)
ids_density_hardness = np.load("extracted_ids_density_hardness.npy", allow_pickle=True)
density = np.load("extracted_density.npy", allow_pickle=True)
hardness = np.load("extracted_hardness.npy", allow_pickle=True)

# remove dash from ids_density_hardness
ids_density_hardness = np.array([s.split('-')[0] for s in ids_density_hardness])

print("\n Chemistry Data")
print("chem shape: ", chem.shape)
print("rruffids_chem shape: ", rruffids_chem.shape)

print("\nNames - density and hardness Data")
print("names shape: ", names_density_hardness.shape)
print("ids shape: ", ids_density_hardness.shape)
print("density shape: ", density.shape)
print("hardness shape: ", hardness.shape)

print("shape of unique ids should match shape of names, ids, density, and hardness: ", np.unique(ids_density_hardness).shape)


count_raw = 0
count_proc = 0

names_raw_all = []
names_proc_all = []
rruffid_raw_all = []
rruffid_proc_all = []

for i in range(chem.shape[0]):
    chemistry = chem[i,:] / 100
    id = rruffids_chem[i].split("-")[0]
    name = names_chem[i]

    # find the index of the id
    idx_density_hardness = np.where(ids_density_hardness == id)[0]

    if idx_density_hardness.size > 0:
        
        # Get the density and the hardness and add to the features (i.e., the chemistry)
        spec_density = density[idx_density_hardness]
        spec_hardness = hardness[idx_density_hardness]
        chemistry = np.append(chemistry, np.array([spec_density, spec_hardness]))

    else: # if no density and hardness for the mineral id, then do not add to training/testing data
        continue

    idx_raw = np.where(rruffids_raw == id)[0]
    idx_proc = np.where(rruffids_proc == id)[0]

    # Loop through the found indexes and build arrays for the spectrum (y_labels) 
    # and the inputs i.e., the chemistry plus the min max x-values (x_inputs)

    # Raw files 
    for j in idx_raw:
        x_temp = x_components_raw[j]
        y_temp = y_components_raw[j]

        if flag_include_min_max:
            minx = x_temp[0] / 1000
            maxx = x_temp[-1] / 1000
            x_input_temp = np.append(chemistry, [minx, maxx])
        else:
            x_input_temp = chemistry

        x_temp = np.expand_dims(x_temp, axis=0)
        y_temp = np.expand_dims(y_temp, axis=0)
        x_input_temp = np.expand_dims(x_input_temp, axis=0)

        names_raw_all.append(name)
        rruffid_raw_all.append(id)
        
        if count_raw == 0:
            y_labels_raw = y_temp  #spectrum to reproduce
            x_inputs_raw = x_input_temp # chemistry, density, and hardness (and min/max wavenumber if flag)
        
        else:
            y_labels_raw = np.concatenate((y_labels_raw, y_temp), axis=0)
            x_inputs_raw = np.concatenate((x_inputs_raw, x_input_temp), axis=0)

        count_raw += 1

    # Processed files 
    for j in idx_proc:
        x_temp = x_components_proc[j]
        y_temp = y_components_proc[j]

        if flag_include_min_max:
            minx = x_temp[0] / 1000
            maxx = x_temp[-1] / 1000
            x_input_temp = np.append(chemistry, [minx, maxx])
        else:
            x_input_temp = chemistry


        x_temp = np.expand_dims(x_temp, axis=0)
        y_temp = np.expand_dims(y_temp, axis=0)
        x_input_temp = np.expand_dims(x_input_temp, axis=0)

        names_proc_all.append(name)
        rruffid_proc_all.append(id)
        
        if count_proc == 0:
            y_labels_proc = y_temp  #spectrum to reproduce
            x_inputs_proc = x_input_temp # chemistry, density, and hardness (and min/max wavenumber if flag)
        
        else:
            y_labels_proc = np.concatenate((y_labels_proc, y_temp), axis=0)
            x_inputs_proc = np.concatenate((x_inputs_proc, x_input_temp), axis=0)

        count_proc += 1


    #if i>10:
    #    break

    if i%100 == 0:
        print("{:.2f}%".format((i / chem.shape[0]) * 100 ))


names_raw_all = np.array(names_raw_all)
rruffid_raw_all = np.array(rruffid_raw_all)

names_proc_all = np.array(names_proc_all)
rruffid_proc_all = np.array(rruffid_proc_all)


print("\nFinal Raw Data shapes")
print("y_labels_raw shape: ", y_labels_raw.shape)
print("x_inputs_raw shape: ", x_inputs_raw.shape)
print("names_raw_all shape: ", names_raw_all.shape)
print("rruffid_raw_all shape: ", rruffid_raw_all.shape)
print("unique rruffid_raw_all shape: ", np.unique(rruffid_raw_all).shape)

print("\nFinal Processed Data shapes")
print("y_labels_proc shape: ", y_labels_proc.shape)
print("x_inputs_proc shape: ", x_inputs_proc.shape)
print("names_proc_all shape: ", names_proc_all.shape)
print("rruffid_proc_all shape: ", rruffid_proc_all.shape)
print("unique rruffid_proc_all shape: ", np.unique(rruffid_proc_all).shape)


# Save numpy files

np.save(savepath + "y_labels_raw.npy", y_labels_raw, allow_pickle=True)
np.save(savepath + "x_inputs_raw.npy", x_inputs_raw, allow_pickle=True)
np.save(savepath + "names_raw.npy", names_raw_all, allow_pickle=True)
np.save(savepath + "rruffid_raw.npy", rruffid_raw_all, allow_pickle=True)

np.save(savepath + "y_labels_proc.npy", y_labels_proc, allow_pickle=True)
np.save(savepath + "x_inputs_proc.npy", x_inputs_proc, allow_pickle=True)
np.save(savepath + "names_proc.npy", names_proc_all, allow_pickle=True)
np.save(savepath + "rruffid_proc.npy", rruffid_proc_all, allow_pickle=True)


# Plot 5 random spectrum from each category
fig_raw, ax_raw = plt.subplots(figsize=(9,5))
ax_raw.set_ylabel("Inensity (normalized)", fontsize=12)
ax_raw.set_xlabel("Wavenumber (1/cm)", fontsize=12)
ax_raw.set_title("Raw Spectrum", fontsize=12)


for i in range(5):
    idx = random.randint(0, len(names_raw_all)-1)
    x = np.arange(1024)
    y = y_labels_raw[idx,:]
    name = names_raw_all[idx]
    chem_input = x_inputs_raw[idx, :]
    ax_raw.plot(xplot, y, label=name)
    #print('{}: '.format(name), chem_input)
ax_raw.legend(loc='best')

fig_proc, ax_proc = plt.subplots(figsize=(9,5))
ax_proc.set_ylabel("Inensity (normalized)", fontsize=12)
ax_proc.set_xlabel("Wavenumber (1/cm)", fontsize=12)
ax_proc.set_title("Processed Spectrum", fontsize=12)
for i in range(5):
    idx = random.randint(0, len(names_proc_all)-1)
    x = np.arange(1024)
    y = y_labels_proc[idx,:]
    name = names_proc_all[idx]
    chem_input = x_inputs_proc[idx, :]
    ax_proc.plot(xplot, y, label=name)
    #print('{}: '.format(name), chem_input)
ax_proc.legend(loc='best')



print("\n-- FINISHED --\n")


plt.show()