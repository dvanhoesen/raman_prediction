import numpy as np
import pandas as pd
from difflib import get_close_matches


# Load the CSV file into a dataframe
file = 'mineral_density.csv'
df = pd.read_csv(file, header=0)

file_names = 'extracted_chemistry_spec_names.npy'
spec_names = np.load(file_names, allow_pickle=True)
spec_names = np.char.lower(spec_names)  # Convert all spec_names to lowercase

remove_chars = " ,!*?)(-#$%^&@"
translation_table = str.maketrans('', '', remove_chars)

spec_ids = np.load('extracted_chemistry_spec_ids.npy')

print("spec_names shape: ", spec_names.shape)
print("unique spec_names shape: ", np.unique(spec_names).shape)
print("spec_ids shahpe: ", spec_ids.shape)

# ANSI escape codes for colors
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"  # Reset color to default

# Function to process the hardness column
def parse_hardness(hardness):
    if pd.isna(hardness) or not isinstance(hardness, str) or hardness.strip() == '':
        return None
    # Check if the value is a single number
    try:
        return float(hardness)
    except ValueError:
        pass
    # Check if the value is a range
    if '-' in hardness:
        try:
            parts = [float(part.strip()) for part in hardness.split('-')]
            if len(parts) == 2:
                return sum(parts) / 2  # Return the middle of the range
        except ValueError:
            pass
    return None  # Return None if no valid number or range is found

# Apply the function to the hardness column
df['hardness'] = df['hardness'].apply(parse_hardness)

# Drop rows with None in hardness or density for plotting
df = df.dropna(subset=['density', 'hardness', 'name'])


names = df.name.values
density = df.density.values
hardness = df.hardness.values

cleaned_names = np.array([name.translate(translation_table) for name in names])
cleaned_names = np.char.lower(cleaned_names)  # Convert all cleaned_names to lowercase


count_in = 0
count_out = 0
names_in_db = []
ids_in_db = []
density_in_db = []
hardness_in_db = []

for i in range(len(cleaned_names)):
    name = cleaned_names[i]
    mineral_density = density[i]
    mineral_hardness = hardness[i]

    if name in spec_names:
        # Print in green if there is a match
        #print(f"{GREEN}{density[i]}\t{hardness[i]}\t{name}{RESET}")

        matching_idxs = np.where(spec_names == name)[0]

        for idx in matching_idxs:

            names_in_db.append(name)
            ids_in_db.append(spec_ids[idx])
            density_in_db.append(mineral_density)
            hardness_in_db.append(mineral_hardness)

            count_in += 1
    
    else:
        # Find the closest match
        #closest_match = get_close_matches(name, spec_names, n=1)
        #if closest_match:
        #    closest = closest_match[0]
        #    print(f"{RED}{density[i]}\t{hardness[i]}\t{name} (Closest match: {closest}){RESET}")
        #else:
        #    print(f"{RED}{density[i]}\t{hardness[i]}\t{name} (No close match found){RESET}")

        count_out += 1

num_unique = np.unique(spec_names)

print("\nMinerals in: ", count_in)
print("Minerals out: ", count_out)
print("Unique minerals: ", len(num_unique))

print("\n")


"""
cnt = 1
for name in spec_names: 
    if name in names_in_db:
        continue
    else: 
        closest_match = get_close_matches(name, cleaned_names, n=1)
        if closest_match:
            print("{} {} - {}".format(cnt, name, closest_match[0]))
        else:
            print("{} {} - {}".format(cnt, name, "None"))
        
        cnt += 1
"""


names_in_db = np.array(names_in_db)
ids_in_db = np.array(ids_in_db)
density_in_db = np.array(density_in_db)
hardness_in_db = np.array(hardness_in_db)

# Find unique IDs and their first occurrence indices
unique_ids, unique_indices = np.unique(ids_in_db, return_index=True)

# Sort indices to maintain original order
unique_indices = np.sort(unique_indices)

# Truncate arrays to keep only unique IDs
names_in_db = names_in_db[unique_indices]
ids_in_db = ids_in_db[unique_indices]
density_in_db = density_in_db[unique_indices]
hardness_in_db = hardness_in_db[unique_indices]


# Save the Numpy arrays
np.save("extracted_names_density_hardness.npy", names_in_db, allow_pickle=True)
np.save("extracted_ids_density_hardness.npy", ids_in_db, allow_pickle=True)
np.save("extracted_density.npy", density_in_db, allow_pickle=True)
np.save("extracted_hardness.npy", hardness_in_db, allow_pickle=True)

