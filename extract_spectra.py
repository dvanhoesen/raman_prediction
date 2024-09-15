import numpy as np
import matplotlib.pyplot as plt
import os, sys
import glob


basepath_raman = 'raman_data' + os.path.sep

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


# Initialize lists to store the names, ids, and x, y components
names = []
rruffids = []
x_components = []
y_components = []

files_failed = []

x_mins = []
x_maxs = []
x_range = []
x_resolution = []

count_file = 0

for file in all_files:
    x_temp = []
    y_temp = []
    
    try:
        with open(file, 'r') as file:
       
            for line in file:
                # Strip any leading or trailing whitespace
                line = line.strip()
                
                # Check for the header lines and extract the required information
                if line.startswith("##NAMES="):
                    names.append(line.split('=')[1])
                
                elif line.startswith("##RRUFFID="):
                    rruffids.append(line.split('=')[1])
                
                # Skip other header lines
                elif line.startswith("##"):
                    continue
                
                # Process the data lines (assuming they start with numbers)
                elif line and line[0].isdigit():
                    data = line.split(',')
                    x_temp.append(float(data[0]))
                    y_temp.append(float(data[1]))
        
    except:
        files_failed.append(file)
        count_file += 1
        continue

    x_temp = np.array(x_temp)
    y_temp = np.array(y_temp)

    min_val = np.amin(x_temp)
    max_val = np.amax(x_temp)
    range_val  = max_val - min_val
    resolution_val = len(x_temp) / range_val

    x_mins.append(min_val)
    x_maxs.append(max_val)
    x_range.append(range_val)
    x_resolution.append(resolution_val)

    #x_components.append(x_temp)
    #y_components.append(y_temp)

    if count_file%100 == 0:
        print("{:.2f}%".format((count_file / len(all_files)) * 100 ))
    
    if count_file > 2000000:
        break

    count_file += 1

# Convert the lists to numpy arrays
names_array = np.array(names)
rruffids_array = np.array(rruffids)
#x_array = np.array(x_components, dtype=object)
#y_array = np.array(y_components, dtype=object)

print("Number of failed files: ", len(files_failed))
for file in files_failed:
    print(file)


## Plot a histogram of the mins and maxs of x
x_mins = np.array(x_mins)
x_maxs = np.array(x_maxs)
x_range = np.array(x_range)
x_resolution = np.array(x_resolution)

bins_mins = np.linspace(0, 350, 50)
bins_maxs = np.linspace(1000, 1600, 50)
bins_range = np.linspace(1000, 1550, 50)
bins_resolution = np.linspace(0.6, 2.1, 50)

fig, ax = plt.subplots(figsize=(9,5))
ax.hist(x_mins, bins=bins_mins, alpha=1, label='mins', color='blue')
ax.hist(x_maxs, bins=bins_maxs, alpha=1, label='maxs', color='red')
ax.set_xlabel('Wavenumber (1/cm)', fontsize=12)
ax.set_ylabel('Spectrum Count', fontsize=12)
ax.legend(loc='best')

fig1, ax1 = plt.subplots(figsize=(9,5))
ax1.hist(x_range, bins=bins_range, alpha=1, label='Wavenumber Range', color='blue')
ax1.set_xlabel('Total Wavenumber (1/cm)', fontsize=12)
ax1.set_ylabel('Spectrum Count', fontsize=12)
ax1.legend(loc='best')

fig2, ax2 = plt.subplots(figsize=(9,5))
ax2.hist(x_resolution, bins=bins_resolution, alpha=1, label='Spectrum Resolution', color='red')
ax2.set_xlabel('Points per Wavenumber', fontsize=12)
ax2.set_ylabel('Spectrum Count', fontsize=12)
ax2.legend(loc='best')


print("Names Array:", names_array)
print("RRUFFID Array:", rruffids_array)
#print("X Components Array:", x_array)
#print("Y Components Array:", y_array)


plt.show()