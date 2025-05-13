import io
import os
import glob
import random
import time


import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageCms
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from controlnet_aux import CannyDetector, HEDdetector
import torch.nn.functional as F
from helpers import generate_1x_sequence, generate_2x_sequence, generate_large_blur_sequence


def unpack_mm_params(p):
    if isinstance(p, (tuple, list)):
        return p[0], p[1]
    elif isinstance(p, (int, float)):
        return p, p
    raise Exception(f'Unknown input parameter type.\nParameter: {p}.\nType: {type(p)}')


def resize_for_crop(image, min_h, min_w):
    img_h, img_w = image.shape[-2:]
    
    if img_h >= min_h and img_w >= min_w:
        coef = min(min_h / img_h, min_w / img_w)
    elif img_h <= min_h and img_w <=min_w:
        coef = max(min_h / img_h, min_w / img_w)
    else:
        coef = min_h / img_h if min_h > img_h else min_w / img_w 

    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def init_controlnet(controlnet_type):
    if controlnet_type in ['canny']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


controlnet_mapping = {
    'canny': CannyDetector,
    'hed': HEDdetector,
}


class BaseClass(Dataset):
    def __init__(
            self, 
            data_dir,
            image_size=(320, 512), 
            hflip_p=0.5,
            controlnet_type='canny',
            split='train',
            *args,
            **kwargs
        ):
        self.split = split
        self.height, self.width = unpack_mm_params(image_size)
        self.data_dir = data_dir
        self.hflip_p = hflip_p
        self.image_size = image_size
        self.length = 0
        
        self.controlnet_processor = init_controlnet(controlnet_type)
        
    def __len__(self):
        return self.length
        

    def load_frames(self, frames):
        # frames: numpy array of shape (N, H, W, C), 0–255
        # → tensor of shape (N, C, H, W) as float
        pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float()
        # normalize to [-1, 1]
        pixel_values = pixel_values / 127.5 - 1.0
        # resize to (self.height, self.width)
        pixel_values = F.interpolate(
            pixel_values,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False
        )
        return pixel_values
        
    def get_batch(self, idx):
        raise Exception('Get batch method is not realized.')

    def __getitem__(self, idx):
        while True:
            try:
                video, caption, motion_blur = self.get_batch(idx)
                break
            except Exception as e:
                print(e)
                idx = random.randint(0, self.length - 1)
            
        video, = [
            resize_for_crop(x, self.height, self.width) for x in [video]
        ] 
        video, = [
            transforms.functional.center_crop(x, (self.height, self.width)) for x in [video]
        ]
        data = {
            'video': video, 
            'caption': caption, 
        }
        return data

def load_as_srgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)

    if 'icc_profile' in img.info:
        icc = img.info['icc_profile']
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst_profile = ImageCms.createProfile("sRGB")

        img = ImageCms.profileToProfile(img, src_profile, dst_profile, outputMode='RGB')
    else:
        img = img.convert("RGB")  # Assume sRGB

    return img
# class CustomControlnetDataset(BaseClass):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.video_paths = glob.glob(os.path.join(self.video_root_dir, '*.mp4'))
#         self.length = len(self.video_paths)
        
#     def get_batch(self, idx):
#         video_path = self.video_paths[idx]
#         caption = os.path.basename(video_path).replace('.mp4', '')
#         pixel_values, controlnet_video = self.load_video_info(video_path)
#         return pixel_values, caption, controlnet_video

class OutsidePhotosDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.*')))
        self.length = len(self.image_paths)

    def __len__(self):
        return self.length
    


        return np.array(img)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        blur_img_original = load_as_srgb(image_path)
        H,W = blur_img_original.size
        blur_img =blur_img_original.resize((self.image_size[1], self.image_size[0])) #cause pil is width, height
        blur_np = np.array([blur_img])

        # Create a black video sequence of same size (window_size = 9)
        window_size = 9
        black_images = np.array([Image.new("RGB", self.image_size, (0, 0, 0)) for _ in range(window_size)])
        intervals = np.array([[i, i+1] for i in range(window_size)])

        pixel_values = self.load_frames(black_images)
        blur_pixel_values = self.load_frames(blur_np)

        file_name = os.path.join("outside_photos", os.path.basename(image_path))

        data = {
            'file_name': file_name,
            "original_size": (H, W),
            'blur_img': torch.tensor(blur_pixel_values, dtype=torch.float32),
            'video': torch.tensor(pixel_values, dtype=torch.float32),
            'caption': "",
            'input_interval': torch.tensor([[0, 9]]),
            'output_interval': torch.tensor(intervals),
        }
        return data
    

class AdobeMotionBlurDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
        if self.split == 'train':
            self.data_dir = os.path.join(self.data_dir, 'train_blur')
        
        elif self.split in ['val', 'test']:
            self.data_dir = os.path.join(self.data_dir, 'test_blur')

        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '*/*_*.png')))#only get files with _w9

        if self.split == "val":
            # self.image_paths = ["/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w09.png",
            #                     "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w09.png"]
            self.image_paths = ["/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w09.png",]
        elif self.split == "test":
            self.image_paths = ["/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0043/00841_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0043/00841_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0043/00841_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0043/00841_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0043/00841_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/GOPR9637b/00137_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/GOPR9637b/00137_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/GOPR9637b/00137_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/GOPR9637b/00137_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/GOPR9637b/00137_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w01.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w03.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w05.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w07.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0179/00577_w09.png"]
                                
        self.length = len(self.image_paths)
        
    def __getitem__(self, idx):

        #get the image path
        image_path = self.image_paths[idx]

        # Extract directory and filename
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)

        # Extract prefix (everything before the first underscore)
        img_id = filename.split("_")[0]
        window_size = int(filename.split("_")[1][1:3]) #get first two letters
        motion_blur_amount = window_size // 2 #half of the window size

        sharp_directory = directory.replace('test_blur', 'full_sharp').replace('train_blur', 'full_sharp')
        sharp_filename = os.path.join(sharp_directory, f"{img_id}.png")

        motion_blur_interval = [0, window_size]

        #override this for now
        motion_blur_amount=4
        window_size = 9
        sharp_image_start = int(img_id) - motion_blur_amount
        sharp_image_end = int(img_id) + motion_blur_amount + 1

        images = []
        intervals = [] #first frame is conditioning frame
        for i in range(0, window_size):
            #convert to 5 digits
            sharp_image_idx = sharp_image_start + i
            current_sharp_image = os.path.join(sharp_directory, f"{str(sharp_image_idx).zfill(5)}.png")
            img = Image.open(current_sharp_image).convert("RGB")
            images.append(img)
            intervals.append([i,i+1])

        while (len(intervals) - 1) % 4 != 0: #this is cause of cogvideo
            #append the last interval to the end of the list and last image
            images.append(images[-1])
            intervals.append([i,i+1])
        
        #add 
        images = np.array(images) #F*H*W*C
        intervals = np.array(intervals) #F*2

        blur_img = np.array([Image.open(image_path).convert("RGB")])
        blur_pixel_values = self.load_frames(blur_img)[:, :, :, :]

        
        pixel_values = self.load_frames(images)[:, :, :, :]
        motion_blur_amount = torch.tensor(motion_blur_amount)

        #get last two parts of the path
        file_name = os.path.join(*image_path.strip(os.sep).split(os.sep)[-2:])

        data = {
            'file_name': file_name,
            'blur_img': blur_pixel_values,
            'video': pixel_values, 
            'caption': "",
            'motion_blur_amount': motion_blur_amount,
            'input_interval': torch.tensor([motion_blur_interval]),
            'output_interval': torch.tensor(intervals),
        }
        return data



class GoProMotionBlurDataset(BaseClass):
    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set blur and sharp directories based on split
        if self.split == 'train':
            self.blur_root = os.path.join(self.data_dir, 'train', 'blur')
            self.sharp_root = os.path.join(self.data_dir, 'train', 'sharp')
        elif self.split in ['val', 'test']:
            self.blur_root = os.path.join(self.data_dir, 'test', 'blur')
            self.sharp_root = os.path.join(self.data_dir, 'test', 'sharp')
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        # Collect all blurred image paths
        pattern = os.path.join(self.blur_root, '*', '*.png')

        self.blur_paths = sorted(glob.glob(pattern))[::-1] #just cause I messed up so lets do reverse to get these images to come first
        if self.split == 'val':
            # Optional: limit validation subset
            self.blur_paths = self.blur_paths[:5]

        # Window and padding parameters
        self.window_size = 7              # original number of sharp frames
        self.pad = 2                      # number of times to repeat last frame
        self.output_length = self.window_size + self.pad
        self.half_window = self.window_size // 2
        self.length = len(self.blur_paths)

        # Normalized input interval: always [-0.5, 0.5]
        self.input_interval = torch.tensor([[-0.5, 0.5]], dtype=torch.float)

        # Precompute normalized output intervals: first for window_size frames, then pad duplicates
        step = 1.0 / (self.window_size - 1)
        # intervals for the original 7 frames
        window_intervals = []
        for i in range(self.window_size):
            start = -0.5 + i * step
            if i < self.window_size - 1:
                end = -0.5 + (i + 1) * step
            else:
                end = 0.5
            window_intervals.append([start, end])
        # append the last interval pad times
        intervals = window_intervals + [window_intervals[-1]] * self.pad
        self.output_interval = torch.tensor(intervals, dtype=torch.float)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Path to the blurred (center) frame
        start_time = time.time()
        blur_path = self.blur_paths[idx]
        seq_name = os.path.basename(os.path.dirname(blur_path))
        frame_name = os.path.basename(blur_path)
        center_idx = int(os.path.splitext(frame_name)[0])

        # Compute sharp frame range [center-half, center+half]
        start_idx = center_idx - self.half_window
        end_idx = center_idx + self.half_window

        # Load sharp frames
        sharp_dir = os.path.join(self.sharp_root, seq_name)
        frames = []
        for i in range(start_idx, end_idx + 1):
            sharp_filename = f"{i:06d}.png"
            sharp_path = os.path.join(sharp_dir, sharp_filename)
            img = Image.open(sharp_path).convert('RGB')
            frames.append(img)

        # Repeat last sharp frame so total frames == output_length
        while len(frames) < self.output_length:
            frames.append(frames[-1])

        # Load blurred image
        blur_img = Image.open(blur_path).convert('RGB')

        # Convert to pixel values via BaseClass loader
        video = self.load_frames(np.array(frames))                    # shape: (output_length, H, W, C)
        blur_input = self.load_frames(np.expand_dims(np.array(blur_img), 0))  # shape: (1, H, W, C)
        end_time = time.time()
        #print(f"Time taken to load and process data: {end_time - start_time:.2f} seconds")
        data = {
            'file_name': os.path.join(seq_name, frame_name),
            'blur_img': blur_input,
            'video': video,
            "caption": "",
            'motion_blur_amount': torch.tensor(self.half_window, dtype=torch.long),
            'input_interval': self.input_interval,
            'output_interval': self.output_interval,
        }
        return data

class FullMotionBlurDataset(BaseClass):
    """
    A dataset that randomly selects among 1×, 2×, or large-blur modes per sample.
    Uses category-specific <split>_list.txt files under each subfolder of FullDataset to assemble sequences.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Assemble sequence directories based on each category's split list
        self.seq_dirs = []
        #if split is trian, then train, else "test"
        text_file_name = "train_list.txt" if self.split == "train" else "test_list.txt"
        for category in sorted(os.listdir(self.data_dir)):
            cat_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            list_file = os.path.join(cat_dir, text_file_name)
            if os.path.isfile(list_file):
                with open(list_file, 'r') as f:
                    for line in f:
                        rel = line.strip()
                        # Skip empty lines
                        if not rel:
                            continue
                        seq_dir = os.path.join(self.data_dir, rel)
                        if os.path.isdir(seq_dir):
                            self.seq_dirs.append(seq_dir)
            else:
                # No list file for this category: fallback to scanning all sequences
                fps_root = os.path.join(cat_dir, 'lower_fps_frames')
                if os.path.isdir(fps_root):
                    for seq in sorted(os.listdir(fps_root)):
                        seq_path = os.path.join(fps_root, seq)
                        if os.path.isdir(seq_path):
                            self.seq_dirs.append(seq_path)
        if self.split == 'val':
            # Optional: limit validation subset
            self.seq_dirs = self.seq_dirs[:5]
        if self.split == 'train':
            #duplicate the training set
            self.seq_dirs = self.seq_dirs * 10

        assert len(self.seq_dirs) > 0, f"No sequences found for split '{self.split}' in {self.data_dir}"

    def __len__(self):
        return len(self.seq_dirs)

    def __getitem__(self, idx):
        seq_dir = self.seq_dirs[idx]
        # gather all raw frames (240fps) for this sequence
        frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
        # pick a mode at random
        mode = random.choice(['1x', '2x', 'large_blur']) #'2x', 'large_blur'])
        #randomly choose base_rate
        
        if mode == '1x' or len(frame_paths) < 50:
            base_rate = random.choice([1, 2])
            blur_img, seq, inp_int, out_int = generate_1x_sequence(
                frame_paths, window_max = 16, output_len=17, base_rate=base_rate)
        elif mode == '2x':
            base_rate = random.choice([1, 2])
            blur_img, seq, inp_int, out_int = generate_2x_sequence(
                frame_paths, window_max = 16, output_len=17, base_rate=base_rate)
        else:
            # compute the maximum integer base_rate such that output_len * base_rate < num_frames
            max_base_rate = (len(frame_paths) - 1) // 17

            #if max_base_rate > 3, then set it to 3
            max_base_rate = min(max_base_rate, 3)
            
            # randomly pick base_rate in [1, max_base_rate]
            base_rate = random.randint(1, max_base_rate)
            
            # randomly choose base_rate such that output_len*base_rate < len(frame_paths)
            blur_img, seq, inp_int, out_int = generate_large_blur_sequence(
                frame_paths, window_max = 16, output_len=17, base_rate=base_rate)

        # Convert to tensors via BaseClass loader
        blur_input = self.load_frames(np.expand_dims(blur_img, 0))
        video = self.load_frames(np.stack(seq, axis=0))

        print("Got video shape: ", video.shape)

        full_fn = os.path.join(seq_dir, frame_paths[0])

        print("input interval: ", inp_int)
        print("output interval: ", out_int)

        data = {
            'file_name': os.path.relpath(full_fn, self.data_dir),
            'blur_img': blur_input,
            'video': video,
            'caption': "",
            'mode': mode,
            'input_interval': inp_int,
            'output_interval': out_int,
        }
        return data
