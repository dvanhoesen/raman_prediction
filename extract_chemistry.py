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
                'ZnO', 'SnO', 'H2O', 'Cr2O3', 'SO3', 'TeO2', 'PbO', 'CuO',
                'FeOT', 'Fe2O3', 'FeO']

normalized_oxides = [name.lower() for name in names_oxides]

average_terms = ['avg', 'average', 'avg.', 'average.', 'average:', 'average: ' 'avg:', 'ave.', 'ave', 'ave.:']
stddev_terms = ['stdev', 'std', 'stddev', 'std dev', 'st dev', 'stnd. dev.', 
                'stnd dev', 's.d.', 'sd', 's d', 's. d.', 'std. dev.:', 'stdev:', 
                'std:', 'stddev:', 'st. dev.', 'std dev.', 'std. dev.', 'stdv', 
                'stdv.', 'st. dv.', 'stdv:', 'st dev.', 'standard deviation',
                'standard dev:']

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

    print("\nFile:", file_count, filename)
    composition = np.zeros(len(names_oxides))
    composition_stddev = np.zeros(len(names_oxides))
    
    # Lowercase the DataFrame for consistent searching
    df_cleaned = df.map(lambda x: str(x).lower() if isinstance(x, str) else x)
    df_lower = df_cleaned.map(lambda x: str(x).replace('*', '') if isinstance(x, str) else x)

    
    # Find the location of the average or standard deviation terms
    # Find the location of the average term paired with a standard deviation term
    avg_loc = None
    flag_avg_found = True

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
        #print('skipped')
        #print(df.head(20))
        #skipped_files.append(file)
        #sys.exit()
        #continue
        avg_row = None
        avg_col = None
        flag_avg_found = False

    for i, oxide in enumerate(normalized_oxides):
        # Find the location of the oxide name
        oxide_loc = np.where(df_lower == oxide)
        if len(oxide_loc[0]) == 0 or len(oxide_loc[1]) == 0:
            continue  # Skip if the oxide is not found

        oxide_row = oxide_loc[0][0]
        oxide_col = oxide_loc[1][0]

        if flag_avg_found and not (oxide_row < avg_row and oxide_col < avg_col):

            # Determine the position of the data based on the relative location of oxide and average
            if oxide_row < avg_row:
                # Data is in the same row as the oxide and in the column of the average
                avg_value = df.iloc[avg_row, oxide_col]
                std_value = df.iloc[avg_row + 1, oxide_col] if avg_row + 1 < df.shape[0] else np.nan
            else:
                if oxide_col >= avg_col:
                    avg_value = None
                    std_value = None
                    #print("MADE IT HERE 1")
                else:
                    # Data is in the same column as the oxide and in the row of the average
                    avg_value = df.iloc[oxide_row, avg_col]
                    std_value = df.iloc[oxide_row, avg_col + 1] if avg_col + 1 < df.shape[1] else np.nan
            
        # If no 'avg' or 'average' found, then assume the row is the data and go until a blank cell + 1
        else:
            avg_value = None
            std_value = None

            # Initialize to start checking after the oxide column
            data_col = oxide_col + 1
            
            # Traverse the row until you find an empty cell followed by the data of interest
            while data_col < df.shape[1]:
                cell_value = df.iloc[oxide_row, data_col]
                
                if pd.isna(cell_value):  # Check if the current cell is empty
                    # Look one cell after the empty cell for the data of interest\
                    try:
                        avg_cell_value = df.iloc[oxide_row, data_col + 1]
                        std_cell_value = df.iloc[oxide_row, data_col + 2]
                        if (pd.notna(avg_cell_value) and isinstance(avg_cell_value, (int, float)) and
                            pd.notna(std_cell_value) and isinstance(std_cell_value, (int, float))):
                                avg_value = avg_cell_value
                                std_value = std_cell_value
                                break
                        else:
                            avg_value = None
                            std_value = None
                            break
                    
                    except Exception as e: 
                        #print("Error: {}".format(e))
                        #print('Oxide: ', oxide)
                        #print(oxide_row, data_col)
                        #sys.exit()
                        avg_value = None
                        std_value = None
                        break

                data_col += 1
            
        # if none found, assume the column immediately to the right is the data with no stddev (no average)
        if avg_value is None:
            right_cell_value = df.iloc[oxide_row, oxide_col + 1]

            if pd.isna(right_cell_value):
                avg_value = 0
                std_value = np.nan
            elif isinstance(right_cell_value, (int, float)):
                avg_value = right_cell_value
                std_value = np.nan
            else:
                print("No average found: ", oxide)
                print(df.iloc[oxide_row].to_string(index=False).replace('\n', ', '))
                print("oxide loc: ", oxide_row, oxide_col)
                print("avg loc: ", avg_row, avg_col)
                sys.exit()

        try:
            print("{}\t{}\t{:.4f}\t{:.4f}".format(i, oxide, avg_value, std_value))

        except Exception as e: 
            print("Error: {}".format(e))
            print("Average: ", avg_value, type(avg_value))
            print("Stddev: ", std_value, type(std_value))
            print("Oxide: ", oxide)
            print("oxide loc: ", oxide_row, oxide_col)
            print("avg loc: ", avg_row, avg_col)
            #sys.exit()
            avg_value = 0
            std_value = np.nan
            continue

        composition[i] = avg_value
        composition_stddev[i] = std_value


    # Append the composition array to the results
    composition = composition
    composition_stddev = composition_stddev
    results.append(composition)
    results_stddev.append(composition_stddev)
    
    #if file_count > 100:
    #    sys.exit()

sample_names = np.array(sample_names)
sample_ids = np.array(sample_ids)

results = np.array(results)
results_stddev = np.array(results_stddev)

print("\nNumber of Samples: ", sample_names.shape, sample_ids.shape, file_count)
print("Results shape: ", results.shape)
print("Results Stddev shape: ", results_stddev.shape)

# Save the Numpy arrays
chem_avg_savename = 'extracted_chemistry_avg.npy'
chem_stddev_savename = 'extracted_chemistry_stddev.npy'
chem_names_savename = 'extracted_chemistry_oxide_names.npy'
chem_spec_names_savename = 'extracted_chemistry_spec_names.npy'
chem_spec_ids_savename = 'extracted_chemistry_spec_ids.npy'

np.save(chem_avg_savename, results, allow_pickle=True)
np.save(chem_stddev_savename, results_stddev, allow_pickle=True)
np.save(chem_names_savename, names_oxides, allow_pickle=True)
np.save(chem_spec_names_savename, sample_names, allow_pickle=True)
np.save(chem_spec_ids_savename, sample_ids, allow_pickle=True)

print("Extracted chemistry avg saved as: ", chem_avg_savename)
print("Extracted chemistry stddev saved as: ", chem_stddev_savename)
print("Extracted chemistry oxide names saved as: ", chem_names_savename)
print("Extracted chemistry spectrum names saved as: ", chem_spec_names_savename)
print("Extracted chemistry spectrum ids saved as: ", chem_spec_ids_savename)

print("\n### FINISHED ###\n\n")


