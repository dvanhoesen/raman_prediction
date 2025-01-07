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
for i in range(len(cleaned_names)):
    name = cleaned_names[i]

    if name in spec_names:
        # Print in green if there is a match
        #print(f"{GREEN}{density[i]}\t{hardness[i]}\t{name}{RESET}")
        names_in_db.append(name)
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

for name in spec_names: 
    if name in names_in_db:
        continue
    else: 
        closest_match = get_close_matches(name, cleaned_names, n=1)
        if closest_match:
            print("{} - {}".format(name, closest_match[0]))
        else:
            print("{} - {}".format(name, "None"))


print("Look at close matches - are there names that should match but do not?")
print("save density and hardness with name that matches the 'extracted_chemistry_spec_names.npy' file")
print("in combine_chem_spectra.py add density and hardness to the chemistry feature array")
print("run leave mineral name out model with density and hardness included as features")
