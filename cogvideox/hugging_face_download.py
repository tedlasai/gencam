from huggingface_hub import snapshot_download

# Download the entire model repository and store it locally
model_path = snapshot_download(repo_id="THUDM/CogVideoX-5b-I2V", cache_dir="./CogVideoX-5b-I2V")

print(f"Model downloaded to: {model_path}")