import os
import glob
import subprocess
from multiprocessing import Pool, Manager, current_process
import argparse
# Config
parser = argparse.ArgumentParser(description="Increase video FPS")
parser.add_argument('--video_dir', type=str, required=True, help='Path to the original videos')
parser.add_argument('--save_dir', type=str, required=True, help='Path to save higher FPS videos')
args = parser.parse_args()
video_dir = args.video_dir
save_dir = args.save_dir
model = "RIFE"
variant = "DR"
checkpoint = "./checkpoints/RIFE/DR-RIFE-pro"
num = "1 1 1 1"
script = "inference_video.py"



# GPU configuration
gpus = [0, 1, 2, 3]

video_paths = []
# Get all .mp4 videos or other video formats in the directory
video_paths = glob.glob(os.path.join(video_dir, "*.mp4"))
video_paths += glob.glob(os.path.join(video_dir, "*.m4v"))
video_paths += glob.glob(os.path.join(video_dir, "*.mov"))
video_paths += glob.glob(os.path.join(video_dir, "*.MOV"))

# Worker function
def run_job(args):
    video_path, gpu_queue = args
    filename = os.path.basename(video_path)

    gpu_id = gpu_queue.get()  # acquire a GPU
    print(f"{current_process().name}: Processing {filename} on GPU {gpu_id}")

    command = [
        "python", script,
        "--video", video_path,
        "--model", model,
        "--variant", variant,
        "--checkpoint", checkpoint,
        "--save_dir", save_dir,
        "--num"
    ] + num.split()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    result = subprocess.run(command, env=env)

    gpu_queue.put(gpu_id)  # release the GPU
    return result.returncode, filename

if __name__ == "__main__":
    with Manager() as manager:
        gpu_queue = manager.Queue()
        for gpu in gpus:
            gpu_queue.put(gpu)

        job_args = [(vp, gpu_queue) for vp in video_paths]

        with Pool(len(gpus)) as pool:
            results = pool.map(run_job, job_args)

        # Check results
        for code, filename in results:
            status = "Success" if code == 0 else "Failed"
            print(f"{filename}: {status}")
