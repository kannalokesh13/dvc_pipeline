import os
import zipfile
import shutil
from pathlib import Path
import glob
from config import upload_blob_file
from config import list_existing_versions



def get_new_version():
    """Get the next version number based on the existing version folders and increment it."""
    existing_version = list_existing_versions()
    return existing_version + 1



def get_file_size(path):
    """Return the size of the file at the specified path."""
    return os.path.getsize(path)


def zip_and_upload(zip_file_path, local_upload_folder=None,version=None):
    """Move the zip file to the local upload folder and upload it."""
    # Move the zip file to the local upload folder
    # shutil.move(zip_file_path, os.path.join(local_upload_folder, os.path.basename(zip_file_path)))

    """ Here we can place code related to the cloud storage and uploading zip file to the cloud"""

    upload_blob_file(zip_file_path,version)

    """ i need to write the code in which the zip file is created in the source folder to remove them"""
    os.remove(zip_file_path)


def process_data_folder(data_folder_path, local_upload_folder=None, max_zip_size_mb=64):
    """Process the data folder, creating zip files and uploading them if they exceed the size limit."""

    # Read and increment the version number
    version = get_new_version()

    sequence_num = 1
    zip_filename = f"{data_folder_path}/data_seq{sequence_num}.zip"
    zipf = zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED)

    # Get all files recursively from the data_folder_path, excluding zip files
    all_files = glob.glob(os.path.join(data_folder_path, '**'), recursive=True)
    all_files = [file for file in all_files if not file.endswith('.zip')]

    for file in all_files:
        # Get relative path to maintain folder structure within the zip file
        arcname = os.path.relpath(file, data_folder_path)
        
        # Add file to the zip file
        zipf.write(file, arcname=arcname)

        # Check the size of the zip file
        if get_file_size(zip_filename) >= max_zip_size_mb * 1024 * 1024:
            # Close current zip file and upload it
            zipf.close()
            zip_and_upload(zip_filename, local_upload_folder, version)

            # Start a new zip file for the remaining data
            sequence_num += 1
            zip_filename = f"{data_folder_path}/data_seq{sequence_num}.zip"
            zipf = zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED)

    # Close and upload the last zip file if it contains any data
    zipf.close()
    if get_file_size(zip_filename) > 0:
        zip_and_upload(zip_filename, local_upload_folder, version)

    print("Successfully uploaded the Data with version:", version)
    return True


if __name__ == "__main__":
    data_folder_path = r"C:\Users\LokeshKanna\Downloads\data\data"
    local_upload_folder = r"C:\Users\LokeshKanna\OneDrive - Matdun Labs India Private Limited\Documents\data_operations\Data_sample"  # Define your local upload folder
    process_data_folder(data_folder_path, local_upload_folder)
