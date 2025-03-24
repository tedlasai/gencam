from huggingface_hub import snapshot_download

# Download the entire model repository and store it locally
model_path = snapshot_download(repo_id="THUDM/CogVideoX1.5-5B-I2V", cache_dir="./CogVideoX1.5-5B-I2V")

print(f"Model downloaded to: {model_path}")