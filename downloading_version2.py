import os
import zipfile
import glob
from pathlib import Path
from config import download_all_blobs

def extract_zip(zip_file_path, output_folder):
    """Extract a zip file to the output folder while preserving the directory structure."""
    with zipfile.ZipFile(zip_file_path, 'r') as zipf:
        zipf.extractall(output_folder)
        
    """ Remove the zip file from the downloaded folder and keep that folder always empty"""    
    os.remove(zip_file_path)

def process_output_folder(output_folder_path, target_folder):
    """Process the output folder, extracting zip files and reconstructing the directory structure."""
    if not os.path.exists(output_folder_path):
        print(f"The output folder {output_folder_path} does not exist.")
        return

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Get all zip files in the output folder, sorted by sequence number in filename
    zip_files = sorted(glob.glob(os.path.join(output_folder_path, '*.zip')), key=lambda x: int(Path(x).stem.split('_seq')[1].split('.')[0]))

    # Extract each zip file
    for zip_file in zip_files:
        print(f"Extracting {zip_file}...")
        extract_zip(zip_file, target_folder)
    
    print("Extraction and reconstruction complete.")

if __name__ == "__main__":

    local_download_folder = r'C:\Users\LokeshKanna\OneDrive - Matdun Labs India Private Limited\Documents\data_operations\azure_downloads'
    download_all_blobs(local_download_folder)

    output_folder_path = r"C:\Users\LokeshKanna\OneDrive - Matdun Labs India Private Limited\Documents\data_operations\azure_downloads"
    target_folder = r"C:\Users\LokeshKanna\OneDrive - Matdun Labs India Private Limited\Documents\data_operations\data_download"
    process_output_folder(output_folder_path, target_folder)
