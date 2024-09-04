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

sample_names = []
sample_ids = []

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
			df = pd.read_excel(file, engine='xlrd')
		elif filename.endswith('.xlsx'):
			df = pd.read_excel(file, engine='openpyxl')
		elif filename.endswith('.ods'):
			df = pd.read_excel(file, engine='odf')


	except Exception as e: 
		print("\nError processing file: {}".format(file))
		print("file count: ", file_count)
		print("Error: {}".format(e))


sample_names = np.array(sample_names)
sample_ids = np.array(sample_ids)

print("Number of Samples: ", sample_names.shape, sample_ids.shape, file_count)
