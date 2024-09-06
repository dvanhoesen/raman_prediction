# Extract chemistry

import numpy as np
import pandas as pd
import sys, os
import glob
import warnings

# Suppress openpyxl xlsx warning
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

basepath = "chemistry_data" + os.path.sep + "Microprobe_Data" + os.path.sep

files_xls = glob.glob(basepath + os.path.sep + "*.xls")
files_xlsx = glob.glob(basepath + os.path.sep + "*.xlsx")
files_ods = glob.glob(basepath + os.path.sep + "*.ods")
files = files_xls + files_xlsx + files_ods

files_bad = [
    "Amarantite__R050153-2__Chemistry__Microprobe_Data_Excel__163.xls"
]

files_bad = [basepath + file for file in files_bad]

# Remove bad files
files = [file for file in files if file not in files_bad]

print("\nXLS files: ", len(files_xls))
print("XLSX files: ", len(files_xlsx))
print("ODS files: ", len(files_ods))
print("Bad files: ", len(files_bad))
print("Number of files after removing bad files: ", len(files))


names_oxides = ['SiO2', 'TiO2', 'Al2O3', 'MgO', 'MnO', 'CaO', 'Na2O', 'K2O', 'P2O5', 
                'ZnO', 'SnO', 'H2O', 'Cr2O3', 'As2O5', 'SO3', 'TeO2', 'PbO', 'CuO',
                'FeOT', 'Fe2O3', 'FeO']

normalized_oxides = [name.lower() for name in names_oxides]

average_terms = ['avg', 'average']
stddev_terms = ['stdev', 'std', 'stddev', 'std dev', 'st dev']

sample_names = []
sample_ids = []
results = []
results_stddev = []
skipped_files = []

file_count = 0
for file in files:
    filename = file.split(os.path.sep)[-1]
    filename_split = filename.split("__")

    sample_name = filename_split[0]
    sample_id = filename_split[1]

    sample_names.append(sample_name)
    sample_ids.append(sample_id)
    
    file_count += 1

    ## Extract Chemistry

    try: 
        # Read the file into a pandas DataFrame based on its extension
        if filename.endswith('.xls'):
            df = pd.read_excel(file, engine='xlrd', header=None)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl', header=None)
        elif filename.endswith('.ods'):
            df = pd.read_excel(file, engine='odf', header=None)
    
    except Exception as e: 
        print("\nError processing file: {}".format(file))
        print("file count: ", file_count)
        print("Error: {}".format(e))
        continue

    print("\nFile:", filename)
    composition = np.zeros(len(names_oxides), dtype=np.float32)
    composition_stddev = np.zeros(len(names_oxides), dtype=np.float32)
    
    # Lowercase the DataFrame for consistent searching
    df_cleaned = df.map(lambda x: str(x).lower() if isinstance(x, str) else x)
    df_lower = df_cleaned.map(lambda x: str(x).replace('*', '') if isinstance(x, str) else x)

    
    # Find the location of the average or standard deviation terms
    # Find the location of the average term paired with a standard deviation term
    avg_loc = None
    for term in average_terms:
        avg_locs = np.where(df_lower == term)
        #print(term, avg_locs)
        for j in range(len(avg_locs[0])):
            avg_row_candidate = avg_locs[0][j]
            avg_col_candidate = avg_locs[1][j]
            
            #print(avg_row_candidate, avg_col_candidate)

            # Check to the right for standard deviation term
            if avg_col_candidate + 1 < df.shape[1] and df_lower.iloc[avg_row_candidate, avg_col_candidate + 1] in stddev_terms:
                #print('check right')
                avg_row = avg_row_candidate
                avg_col = avg_col_candidate
                avg_loc = (avg_row, avg_col)
                break

            # Check below for standard deviation term
            if avg_row_candidate + 1 < df.shape[0] and df_lower.iloc[avg_row_candidate + 1, avg_col_candidate] in stddev_terms:
                #print('check left')
                avg_row = avg_row_candidate
                avg_col = avg_col_candidate
                avg_loc = (avg_row, avg_col)
                break

        if avg_loc is not None:
            break  # Stop searching once a valid pair is found


    if avg_loc is None: # no average cell found
        print('skipped')
        print(df.head(20))
        skipped_files.append(file)
        sys.exit()
        continue  

    for i, oxide in enumerate(normalized_oxides):
        # Find the location of the oxide name
        oxide_loc = np.where(df_lower == oxide)
        if len(oxide_loc[0]) == 0 or len(oxide_loc[1]) == 0:
            continue  # Skip if the oxide is not found

        oxide_row = oxide_loc[0][0]
        oxide_col = oxide_loc[1][0]


        # Determine the position of the data based on the relative location of oxide and average
        if oxide_row < avg_row:
            # Data is in the same row as the oxide and in the column of the average
            avg_value = df.iloc[avg_row, oxide_col]
            std_value = df.iloc[avg_row + 1, oxide_col] if avg_row + 1 < df.shape[0] else np.nan
        else:
            # Data is in the same column as the oxide and in the row of the average
            avg_value = df.iloc[oxide_row, avg_col]
            std_value = df.iloc[oxide_row, avg_col + 1] if avg_col + 1 < df.shape[1] else np.nan
        
        print(avg_value, type(avg_value))
        print(std_value, type(std_value))
        print('messing up because as2o5 and as2o3 listed above/below the average, std text
        print("{}\t{}\t{:.4f}\t{:.4f}".format(i, oxide, avg_value, std_value))
        composition[i] = avg_value
        composition_stddev[i] = std_value


    # Append the composition array to the results
    composition = composition.astype(np.float32)
    composition_stddev = composition_stddev.astype(np.float32)
    results.append(composition)
    results_stddev.append(composition_stddev)

    #print(composition)
    
    if file_count > 20:
        sys.exit()

sample_names = np.array(sample_names)
sample_ids = np.array(sample_ids)

print("Number of Samples: ", sample_names.shape, sample_ids.shape, file_count)
