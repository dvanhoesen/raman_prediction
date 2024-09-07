import os, sys
import requests
import zipfile
from urllib.parse import urlparse

# URLs to download and extract
urls_raman = [
    'https://rruff.info/zipped_data_files/raman/LR-Raman.zip',
    'https://rruff.info/zipped_data_files/raman/excellent_oriented.zip',
    'https://rruff.info/zipped_data_files/raman/excellent_unoriented.zip',
    'https://rruff.info/zipped_data_files/raman/fair_oriented.zip',
    'https://rruff.info/zipped_data_files/raman/fair_unoriented.zip',
    'https://rruff.info/zipped_data_files/raman/ignore_unoriented.zip',
    'https://rruff.info/zipped_data_files/raman/poor_oriented.zip',
    'https://rruff.info/zipped_data_files/raman/poor_unoriented.zip',
    'https://rruff.info/zipped_data_files/raman/unrated_oriented.zip',
    'https://rruff.info/zipped_data_files/raman/unrated_unoriented.zip'
]

# uncomment to only extract the microprobe chemistry data
#urls_raman = []

urls_chemistry = [
    'https://rruff.info/zipped_data_files/chemistry/Microprobe_Data.zip'
]


def download_and_extract_zip(url, output_dir):

    print("\nDownloading {}".format(url))

    # Parse the URL to get the filename
    parsed_url = urlparse(url)
    zip_filename = os.path.basename(parsed_url.path)
    
    # Download the ZIP file
    response = requests.get(url)
    zip_filepath = os.path.join(output_dir, zip_filename)

    # Create a subfolder for the zip contents
    unzip_dir = os.path.join(output_dir, os.path.splitext(zip_filename)[0])
    os.makedirs(unzip_dir, exist_ok=True)
    
    # Write the downloaded ZIP file to disk
    with open(zip_filepath, 'wb') as zip_file:
        zip_file.write(response.content)

    # Extract the ZIP file into the output directory
    print("Saving {} to {}".format(zip_filename, output_dir))
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    
    # Delete the ZIP file after extraction
    print("Deleting zip file: ", zip_filepath)
    os.remove(zip_filepath)


# Create the directories for the Raman and Chemistry data
output_dir_raman = 'raman_data'
output_dir_chemistry = 'chemistry_data'

os.makedirs(output_dir_raman, exist_ok=True)
os.makedirs(output_dir_chemistry, exist_ok=True)

# Loop through each URL, download, extract, and delete the ZIP file
for url in urls_raman:
    download_and_extract_zip(url, output_dir_raman)

for url in urls_chemistry:
    download_and_extract_zip(url, output_dir_chemistry)

print("\n### FINISHED ###\n\n")

