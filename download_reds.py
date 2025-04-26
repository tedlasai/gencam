from huggingface_hub import hf_hub_download
import shutil
import os

# Define the repository ID and filenames
repo_id = "snah/REDS"
filenames = ["train_sharp.zip", "val_sharp.zip"]

# Create a directory to store the downloaded files
os.makedirs("REDS_dataset", exist_ok=True)

# Download each file and move it to the desired directory
for filename in filenames:
    # Download the file; it will be cached locally
    cached_file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    
    # Define the destination path
    destination_path = os.path.join("REDS_dataset", filename)
    
    # Copy the file from the cache to the destination directory
    shutil.copy(cached_file_path, destination_path)
    
    print(f"Downloaded and saved {filename} to {destination_path}")
