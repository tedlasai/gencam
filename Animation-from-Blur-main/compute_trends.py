import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, set_start_method, Queue
import torch


# Replace this with your actual import
# from your_module import compute_quantized_flow


import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large
from PIL import Image

def compute_quantized_flow(imgs, threshold=0.25, device=None):
    """
    Compute averaged flow across adjacent images, threshold small magnitudes, 
    and quantize into 4 diagonal bins.

    Args:
        images:
        threshold    (float):   drop any pixel whose flow magnitude < threshold.
        device       (torch.device or None):  
                              where to run RAFT; if None we pick CUDA if available.

    Returns:
        flow_avg  (np.ndarray[2,H,W]): the mean flow field (after zeroing small vectors).
        quant_img (np.ndarray[H,W,3], dtype=uint8): color image with:
                      [-1,-1]→green, [ 1,-1]→red, [-1, 1]→yellow, [ 1, 1]→blue, zeros→black.
    """
    with torch.no_grad():
        # pick device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load RAFT
        model = raft_large(pretrained=True).to(device).eval()


        def flow_pair(im1_np, im2_np):
            im1 = TF.to_tensor(im1_np).unsqueeze(0).to(device)
            im2 = TF.to_tensor(im2_np).unsqueeze(0).to(device)
            flows = model(im1, im2)
            return flows[-1][0].detach().cpu().numpy()  # 2×H×W


        # compute and average flows
        all_flows = [flow_pair(imgs[i], imgs[i+1]) for i in range(len(imgs)-1)]
        flow_avg = np.mean(all_flows, axis=0)  # 2×H×W

        # zero out small motions
        u, v = flow_avg
        mag = np.sqrt(u*u + v*v)
        small = mag < threshold
        u[small] = 0
        v[small] = 0
        flow_avg = np.stack([u, v], axis=0)

        # quantize signs
        su = np.sign(u).astype(int)
        sv = np.sign(v).astype(int)

        # pack into a single array
        quant_flow = np.stack([su, sv], axis=0)  # shape (2, H, W)

        # build viz image
        H, W = su.shape
        quant_img = np.zeros((H, W, 3), dtype=np.uint8)
        quant_img[(su==-1)&(sv==-1)] = [0, 255,   0]  # green  for bottom-left
        quant_img[(su== 1)&(sv==-1)] = [255, 0,   0]  # red    for bottom-right
        quant_img[(su==-1)&(sv== 1)] = [255,255,  0]  # yellow for top-left
        quant_img[(su== 1)&(sv== 1)] = [0,   0, 255]  # blue   for top-right
        # zeros remain black
        torch.cuda.empty_cache()

    return flow_avg, quant_img

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = plt.imread(path)
        images.append(img)
    return np.array(images)

def worker(device_id, task_queue):
    torch.cuda.set_device(device_id)

    while not task_queue.empty():
        try:
            prefix, paths, trend_dir = task_queue.get_nowait()

            paths = sorted(paths)[:7]
            images = load_images(paths)

            quant_flow, quant_viz = compute_quantized_flow(images, threshold=0.25, device=f"cuda:{device_id}")

            npy_path = os.path.join(trend_dir, f"{prefix}_trend.npy")
            png_path = os.path.join(trend_dir, f"{prefix}_trend.png")

            np.save(npy_path, quant_flow)
            plt.imsave(png_path, quant_viz)
            print(f"[GPU {device_id}] Processed {prefix}: saved to {npy_path} and {png_path}")

        except Exception as e:
            print(f"[GPU {device_id}] Error processing {prefix}: {e}")

def process_directory_with_gpus(base_path, num_gpus=4):
    sharp_dir = os.path.join(base_path, "sharp")
    trend_dir = os.path.join(base_path, "trend")
    os.makedirs(trend_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(sharp_dir, "*.png")))

    prefix_dict = defaultdict(list)
    for path in image_paths:
        filename = os.path.basename(path)
        prefix = filename.split("_")[0]
        prefix_dict[prefix].append(path)

    tasks = [(prefix, paths, trend_dir) for prefix, paths in prefix_dict.items() if len(paths) >= 7]

    # Setup multiprocessing
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)

    processes = []
    for device_id in range(num_gpus):
        p = Process(target=worker, args=(device_id, task_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    try:
        set_start_method('spawn')  # Safe for PyTorch + CUDA
    except RuntimeError:
        pass

    root_dir = "/home/tedlasai/genCamera/Animation-from-Blur-main/dataset/gopro"
    subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for subdir in subdirs:
        print(f"Processing {subdir}")
        process_directory_with_gpus(subdir, num_gpus=4)