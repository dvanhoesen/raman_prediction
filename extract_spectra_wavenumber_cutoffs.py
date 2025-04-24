import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob
import random


basepath_raman = 'raman_data' + os.path.sep

low_wavenumber_cutoff = 50
high_wavenumber_cutoff = 1700
interpolation_points = 1024

# URLs to download and extract
folders_raman = [
    'excellent_oriented',
    'excellent_unoriented',
    'fair_oriented',
    'fair_unoriented',
]

all_files = []

for folder in folders_raman:
    basepath = basepath_raman + folder + os.path.sep
    files = glob.glob(basepath + "*.txt")
    print("Number of files in {}: {}".format(folder, len(files)))

    all_files += files

print("Total number of files: ", len(all_files))

files_raw = []
files_processed = []
for file in all_files:
    if "_raw" in file.lower():
        files_raw.append(file)
    if "_processed" in file.lower():
        files_processed.append(file)


print("Raw files: ", len(files_raw))
print("Processed files: ", len(files_processed))
print("raw + processed files: ", len(files_raw) + len(files_processed))


# Initialize lists to store the names, ids, and x, y components
names_raw = []
rruffids_raw = []
x_mins_raw = []
x_maxs_raw = []

names_proc = []
rruffids_proc = []
x_mins_proc = []
x_maxs_proc = []


min_wavenumber_proc = []
max_wavenumber_proc = []

files_failed = []
files_exception = []


count_file = 0
count_raw_files = 0
count_proc_files = 0

for file in all_files:
    x_temp = []
    y_temp = []

    
    if "_raw" in file.lower():
        label = 'raw'
    if "_processed" in file.lower():
        label = 'proc'
    
    try:
        with open(file, 'r', encoding='utf-8') as f:
       
            for line in f:
                # Strip any leading or trailing whitespace
                line = line.strip()
                
                # Check for the header lines and extract the required information
                if line.startswith("##NAMES="):
                    name = line.split('=')[1]
                    if label=='raw': 
                        names_raw.append(name)
                    if label =='proc':
                        names_proc.append(name)
                
                elif line.startswith("##RRUFFID="):
                    rruffid = line.split('=')[1]
                    if label=='raw': 
                        rruffids_raw.append(rruffid)
                    if label=='proc': 
                        rruffids_proc.append(rruffid)

                elif line.startswith("##END"):
                    break
                
                # Skip other header lines
                elif line.startswith("##"):
                    continue
                
                # Process the data lines (assuming they start with numbers)
                elif line and line[0].isdigit():
                    data = line.split(',')
                    x_temp.append(float(data[0]))
                    y_temp.append(float(data[1]))
        
    except Exception as e:
        files_failed.append(file)
        files_exception.append(e)
        count_file += 1
        continue
    

    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)

    min_val = np.amin(x_temp)
    max_val = np.amax(x_temp)


    if label=='proc':
        min_wavenumber_proc.append(min_val)
        max_wavenumber_proc.append(max_val)


    idx_min = np.argmin(x_temp)
    idx_max = np.argmax(x_temp) - len(x_temp)

    fv = x_temp[0]
    lv = x_temp[-1]
    vd = lv-fv
    ftv = x_temp[1] - x_temp[0]
    
    # The spectrum is decreasing in order
    if vd < 0 or (idx_min > 1 or idx_max < -1):
        x_temp = x_temp[::-1]
        y_temp = y_temp[::-1]

        # recalculate the values
        min_val = np.amin(x_temp)
        max_val = np.amax(x_temp)

        idx_min = np.argmin(x_temp)
        idx_max = np.argmax(x_temp) - len(x_temp)

        fv = x_temp[0]
        lv = x_temp[-1]
        vd = lv-fv
        ftv = x_temp[1] - x_temp[0]
    

    if label=='raw': 
        x_mins_raw.append(min_val)
        x_maxs_raw.append(max_val)
    if label=='proc': 
        x_mins_proc.append(min_val)
        x_maxs_proc.append(max_val)
    

    # Interpolate the x and y values to a numpy array of shape (interpolation_points,)
    new_x = np.linspace(low_wavenumber_cutoff, high_wavenumber_cutoff, interpolation_points, endpoint=True)
    
    # Set up interpolation, filling out-of-bounds values with zero
    #new_y = np.interp(new_x, x_temp, y_temp)
    new_y = np.interp(new_x, x_temp, y_temp, left=0, right=0)

    new_x = np.expand_dims(new_x, axis=0)
    new_y = np.expand_dims(new_y, axis=0)

    # Normalize by max
    new_y = new_y / np.amax(new_y)
    
    if label=='raw': 
        if count_raw_files == 0:
            x_components_raw = new_x
            y_components_raw = new_y
        else:
            x_components_raw = np.concatenate((x_components_raw, new_x), axis=0)
            y_components_raw = np.concatenate((y_components_raw, new_y), axis=0)
        
        count_raw_files += 1
    
    if label=='proc': 
        if count_proc_files == 0:
            x_components_proc = new_x
            y_components_proc = new_y
        else:
            x_components_proc = np.concatenate((x_components_proc, new_x), axis=0)
            y_components_proc = np.concatenate((y_components_proc, new_y), axis=0)
        
        count_proc_files += 1


    if count_file%100 == 0:
        print("{:.2f}%".format((count_file / len(all_files)) * 100 ))

    count_file += 1

    #if count_file > 1000:
    #    break

# Convert the lists to numpy arrays
names_raw = np.array(names_raw)
rruffids_raw = np.array(rruffids_raw)
x_mins_raw = np.array(x_mins_raw)
x_maxs_raw = np.array(x_maxs_raw)

names_proc = np.array(names_proc)
rruffids_proc = np.array(rruffids_proc)
x_mins_proc = np.array(x_mins_proc)
x_maxs_proc = np.array(x_maxs_proc)


basepath = "extracted_data_wavenumber_cutoffs" + os.path.sep

min_wavenumber_proc = np.array(min_wavenumber_proc)
max_wavenumber_proc = np.array(max_wavenumber_proc)

print("proc min_wavenumber array shape: ", min_wavenumber_proc.shape)
print("proc max_wavenumber array shape: ", max_wavenumber_proc.shape)

np.save(basepath + "wavenumber_mins.npy", min_wavenumber_proc, allow_pickle=True)
np.save(basepath + "wavenumber_maxs.npy", max_wavenumber_proc, allow_pickle=True)

sys.exit()


print("Number of failed files: ", len(files_failed))
for i in range(len(files_failed)):
    print(files_failed[i])
    print(files_exception[i])


print("\nRaw Files")
print("names_raw shape:", names_raw.shape)
print("rruffids_raw shape:", rruffids_raw.shape)
print("x_components_raw shape:", x_components_raw.shape)
print("y_components_raw shape:", y_components_raw.shape)
print("x_mins_raw shape: ", x_mins_raw.shape)
print("x_maxs_raw shape: ", x_maxs_raw.shape)

print("\nProcessed Files")
print("names_proc shape:", names_proc.shape)
print("rruffids_proc shape:", rruffids_proc.shape)
print("x_components_proc shape:", x_components_proc.shape)
print("y_components_proc shape:", y_components_proc.shape)
print("x_mins_proc shape: ", x_mins_proc.shape)
print("x_maxs_proc shape: ", x_maxs_proc.shape)

# Save the Numpy arrays

np.save(basepath + "names_raw.npy", names_raw, allow_pickle=True)
np.save(basepath + "rruffids_raw.npy", rruffids_raw, allow_pickle=True)
np.save(basepath + "x_components_raw.npy", x_components_raw, allow_pickle=True)
np.save(basepath + "y_components_raw.npy", y_components_raw, allow_pickle=True)

np.save(basepath + "names_proc.npy", names_proc, allow_pickle=True)
np.save(basepath + "rruffids_proc.npy", rruffids_proc, allow_pickle=True)
np.save(basepath + "x_components_proc.npy", x_components_proc, allow_pickle=True)
np.save(basepath + "y_components_proc.npy", y_components_proc, allow_pickle=True)

print("\n-- FINISHED --\n")


# Plot 5 random spectrum from each category
fig_raw, ax_raw = plt.subplots(figsize=(9,5))
ax_raw.set_ylabel("Inensity (normalized)", fontsize=12)
ax_raw.set_xlabel("Wavenumber (1/cm)", fontsize=12)
ax_raw.set_title("Raw Spectrum", fontsize=12)
for i in range(5):
    idx = random.randint(0, len(names_raw))
    x = x_components_raw[idx,:]
    y = y_components_raw[idx,:]
    rruff_id = rruffids_raw[idx]
    name = names_raw[idx]
    ax_raw.plot(x, y, label='{}-{}'.format(rruff_id, name))
ax_raw.legend(loc='best')

fig_proc, ax_proc = plt.subplots(figsize=(9,5))
ax_proc.set_ylabel("Inensity (normalized)", fontsize=12)
ax_proc.set_xlabel("Wavenumber (1/cm)", fontsize=12)
ax_proc.set_title("Processed Spectrum", fontsize=12)
for i in range(5):
    idx = random.randint(0, len(names_proc))
    x = x_components_proc[idx,:]
    y = y_components_proc[idx,:]
    rruff_id = rruffids_proc[idx]
    name = names_proc[idx]
    ax_proc.plot(x, y, label='{}-{}'.format(rruff_id, name))
ax_proc.legend(loc='best')

plt.show()