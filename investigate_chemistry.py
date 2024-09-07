import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Extract data from Numpy files
chem_avg_savename = 'extracted_chemistry_avg.npy'
chem_stddev_savename = 'extracted_chemistry_stddev.npy'
chem_names_savename = 'extracted_chemistry_oxide_names.npy'
chem_spec_names_savename = 'extracted_chemistry_spec_names.npy'
chem_spec_ids_savename = 'extracted_chemistry_spec_ids.npy'


avgs = np.load(chem_avg_savename, allow_pickle=True)
stddevs = np.load(chem_stddev_savename, allow_pickle=True)
names_oxides = np.load(chem_names_savename, allow_pickle=True)
names = np.load(chem_spec_names_savename, allow_pickle=True)
ids = np.load(chem_spec_ids_savename, allow_pickle=True)


print("averages shape: ", avgs.shape)
print("stddevs shape: ", stddevs.shape)
print("oxide names shape: ", names_oxides.shape)
print("spectrum names shape: ", names.shape)
print("spectrum ids shape: ", ids.shape)

# Look at the sums of the oxides to see if any greater than 100 or close to 0
total_wt_percent = np.sum(avgs, axis=1)
idxs_gt_100 = np.where(total_wt_percent > 101)
spec_wt_percent_gt_100 = total_wt_percent[idxs_gt_100]
spec_names_gt_100 = names[idxs_gt_100]
spec_ids_gt_100 = ids[idxs_gt_100]

print("\n{}\t{}\t{}".format("weight %", "sample id", "sample name"))
for i in range(len(spec_names_gt_100)):
    spec_name = spec_names_gt_100[i]
    spec_id = spec_ids_gt_100[i]
    spec_wt_percent = spec_wt_percent_gt_100[i]

    print("{:.4f}\t{}  \t{}".format(spec_wt_percent, spec_id, spec_name))

sys.exit()

fig, ax = plt.subplots()
ax.hist(total_wt_percent, bins=25)
plt.show()

#for i in range(len(names_oxides)):
#    fig, ax = plt.subplots()
#    ax.