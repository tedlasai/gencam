import os
import sys
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel
from PIL import Image

# Load model
model_id = "/datasets/sai/gencam/cogvideox/CogVideoX-5b-I2V/models--THUDM--CogVideoX-5b-I2V/snapshots/a6f0f4858a8395e7429d82493864ce92bf73af11"

transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder,
    transformer=transformer,
    vae=vae,
    torch_dtype=torch.bfloat16,
)

pipe.to(device='cuda')
pipe.vae.enable_tiling()

def main(image_path):
    if not os.path.isfile(image_path):
        print(f"Error: File not found: {image_path}")
        return

    image = load_image(image_path)
    print("Image size before resize:", image.size)
    image = image.resize((1360, 768), resample=Image.BILINEAR)
    print("Image size after resize:", image.size)

    prompt = "A soccer ball moving. High quality no artifacts, realistic, realistic motion, consistent motion, high detail, no distortions, ultra realistic"

    with torch.no_grad():
        video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50, num_frames=49).frames[0]

    # Create output path
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    export_to_video(video, output_path, fps=8)
    print(f"Saved video to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
    else:
        main(sys.argv[1])
