import io
import os
import glob
from pathlib import Path
import pickle
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
from helpers import generate_1x_sequence, generate_2x_sequence, generate_large_blur_sequence, generate_test_case


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
            "num_frames": self.window_size,
            "mode": "1x",
        }
        return data
class OutsidePhotosDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '*.*')))
        INTERVALS = [{"in_start": 0, "in_end": 4, "out_start": 0, "out_end": 4, "center": 2, "window_size": 4, "mode": "1x", "fps": 240},
                    {"in_start": 0, "in_end": 8, "out_start": 0, "out_end": 8, "center": 4, "window_size": 8, "mode": "1x", "fps": 240},
                    {"in_start": 0, "in_end": 12, "out_start": 0, "out_end": 12, "center": 6, "window_size": 12, "mode": "1x", "fps": 240},
                    {"in_start": 0, "in_end": 16, "out_start": 0, "out_end": 16, "center": 8, "window_size": 16, "mode": "1x", "fps": 240},
                    {"in_start": 4, "in_end": 12, "out_start": 0, "out_end": 16, "center": 8, "window_size": 16, "mode": "2x", "fps": 240},
                    {"in_start": 0, "in_end": 32, "out_start": 0, "out_end": 32, "center": 12, "window_size": 32, "mode": "lb", "fps": 120},
                    {"in_start": 0, "in_end": 48, "out_start": 0, "out_end": 48, "center": 24, "window_size": 48, "mode": "lb", "fps": 80}]
                    

        self.cleaned_intervals = []
        for image_path in self.image_paths:
            for interval in INTERVALS:
                #create a copy of the interval dictionary
                i = interval.copy()
                #add the image path to the interval dictionary
                i['video_name'] = image_path #FIX THIS
                video_name = i['video_name']
                in_start  = i['in_start']
                in_end    = i['in_end']
                out_start = i['out_start']
                out_end   = i['out_end']
                center    = i['center']
                window    = i['window_size']
                mode      = i['mode']
                fps       = i['fps'] # e.g. "lower_fps_frames/720p_240fps_1/frame_00247.png"

                # Get base directory and frame prefix
                vid_name_w_extension = os.path.basename(video_name).split('.')[0]

                output_name = (
                    f"{vid_name_w_extension}_"
                    f"in{in_start:04d}_ie{in_end:04d}_"
                    f"os{out_start:04d}_oe{out_end:04d}_"
                    f"ctr{center:04d}_win{window:04d}_"
                    f"fps{fps:04d}_{mode}.mp4"
                )

                full_output_path = os.path.join("/datasets/sai/gencam/cogvideox/training/cogvideox-outsidephotos/deblurred", output_name) #THIS IS A HACK - YOU NEED TO UPDATE THIS TO YOUR OUTPUT DIRECTORY

                # Keep only if output doesn't exist
                if not os.path.exists(full_output_path):
                    self.cleaned_intervals.append(i)


        self.length = len(self.cleaned_intervals)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):

        interval = self.cleaned_intervals[idx]

        in_start  = interval['in_start']
        in_end    = interval['in_end']
        out_start = interval['out_start']
        out_end   = interval['out_end']
        center = interval['center']
        window = interval['window_size']
        mode   = interval['mode']
        fps    = interval['fps']


        image_path = interval['video_name']
        blur_img_original = load_as_srgb(image_path)
        H,W = blur_img_original.size



        frame_paths = []
        #replicate frame_paths to 100 frames
        frame_paths = ["/datasets/sai/gencam/soccerballs/still.png" for _ in range(window)] #any random path replicated

        # generate test case
        _, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
            frame_paths=frame_paths, window_max=window, in_start=in_start, in_end=in_end, out_start=out_start,out_end=out_end, center=center, mode=mode, fps=fps
        )
        file_name = image_path

        # Get base directory and frame prefix
        relative_file_name = os.path.relpath(file_name, self.data_dir)
        base_dir = os.path.dirname(relative_file_name)
        frame_stem = os.path.splitext(os.path.basename(file_name))[0]  # "frame_00000"
        # Build new filename
        new_filename = (
            f"{frame_stem}_"
            f"in{in_start:04d}_ie{in_end:04d}_"
            f"os{out_start:04d}_oe{out_end:04d}_"
            f"ctr{center:04d}_win{window:04d}_"
            f"fps{fps:04d}_{mode}.png"
        )

        blur_img =blur_img_original.resize((self.image_size[1], self.image_size[0])) #cause pil is width, height

        # Final path
        relative_file_name = os.path.join(base_dir, new_filename)


        blur_input = self.load_frames(np.expand_dims(blur_img, 0).copy())
        # seq_frames is list of frames; stack along time dim
        video = self.load_frames(np.stack(seq_frames, axis=0))


        data = {
            'file_name': relative_file_name,
            "original_size": (H, W),
            'blur_img': blur_input,
            'video': video,
            'caption': "",
            'input_interval': inp_int,
            'output_interval': out_int,
            "num_frames": num_frames,
        }
        return data
    
class FullMotionBlurDataset(BaseClass):
    """
    A dataset that randomly selects among 1×, 2×, or large-blur modes per sample.
    Uses category-specific <split>_list.txt files under each subfolder of FullDataset to assemble sequences.
    In 'test' split, it instead loads precomputed intervals from intervals_test.pkl and uses generate_test_case.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_dirs = []

        # TEST split: load fixed intervals early
        if self.split == 'test':
            pkl_path = os.path.join(self.data_dir, 'intervals_test.pkl')
            with open(pkl_path, 'rb') as f:
                self.test_intervals = pickle.load(f)
            assert self.test_intervals, f"No test intervals found in {pkl_path}"
            
            cleaned_intervals = []
            for interval in self.test_intervals:
                # Extract interval values
                in_start  = interval['in_start']
                in_end    = interval['in_end']
                out_start = interval['out_start']
                out_end   = interval['out_end']
                center    = interval['center']
                window    = interval['window_size']
                mode      = interval['mode']
                fps       = interval['fps'] # e.g. "lower_fps_frames/720p_240fps_1/frame_00247.png"
                category, seq = interval['video_name'].split('/')#.split('/')
                seq_dir = os.path.join(self.data_dir, category, 'lower_fps_frames', seq)
                frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
                rel_path = os.path.relpath(frame_paths[center], self.data_dir)
                rel_path = os.path.splitext(rel_path)[0] # remove the file extension
            # #print("File name: ", file_name)                
                # Construct filename
                output_name = (
                    f"{rel_path}_"
                    f"in{in_start:04d}_ie{in_end:04d}_"
                    f"os{out_start:04d}_oe{out_end:04d}_"
                    f"ctr{center:04d}_win{window:04d}_"
                    f"fps{fps:04d}_{mode}.mp4"
                )
                full_output_path = os.path.join("/datasets/sai/gencam/cogvideox/training/cogvideox-full-sample/deblurred", output_name) #THIS IS A HACK - YOU NEED TO UPDATE THIS TO YOUR OUTPUT DIRECTORY

                # Keep only if output doesn't exist
                if not os.path.exists(full_output_path):
                    cleaned_intervals.append(interval)
            print("Len of test intervals after cleaning: ", len(cleaned_intervals))
            print("Len of test intervals before cleaning: ", len(self.test_intervals))
            self.test_intervals = cleaned_intervals


        # TRAIN/VAL: build seq_dirs from each category's list or fallback
        list_file = 'train_list.txt' if self.split == 'train' else 'test_list.txt'
        for category in sorted(os.listdir(self.data_dir)):
            cat_dir = os.path.join(self.data_dir, category)
            if not os.path.isdir(cat_dir):
                continue
            list_path = os.path.join(cat_dir, list_file)
            if os.path.isfile(list_path):
                with open(list_path, 'r') as f:
                    for line in f:
                        rel = line.strip()
                        if not rel:
                            continue
                        seq_dir = os.path.join(self.data_dir, rel)
                        if os.path.isdir(seq_dir):
                            self.seq_dirs.append(seq_dir)
            else:
                fps_root = os.path.join(cat_dir, 'lower_fps_frames')
                if os.path.isdir(fps_root):
                    for seq in sorted(os.listdir(fps_root)):
                        seq_path = os.path.join(fps_root, seq)
                        if os.path.isdir(seq_path):
                            self.seq_dirs.append(seq_path)

        if self.split == 'val':
            self.seq_dirs = self.seq_dirs[:5]
        if self.split == 'train':
            self.seq_dirs *= 10

        assert self.seq_dirs, \
            f"No sequences found for split '{self.split}' in {self.data_dir}"

    def __len__(self):
        return len(self.test_intervals) if self.split == 'test' else len(self.seq_dirs)

    def __getitem__(self, idx):
        # Prepare base items
        if self.split == 'test':
            interval = self.test_intervals[idx]
            category, seq = interval['video_name'].split('/')
            seq_dir = os.path.join(self.data_dir, category, 'lower_fps_frames', seq)
            frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))

            in_start  = interval['in_start']
            in_end    = interval['in_end']
            out_start = interval['out_start']
            out_end   = interval['out_end']
            center = interval['center']
            window = interval['window_size']
            mode   = interval['mode']
            fps    = interval['fps']

            # generate test case
            blur_img, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
                frame_paths=frame_paths, window_max=window, in_start=in_start, in_end=in_end, out_start=out_start,out_end=out_end, center=center, mode=mode, fps=fps
            )
            file_name = frame_paths[center]
            
        else:
            seq_dir = self.seq_dirs[idx]
            frame_paths = sorted(glob.glob(os.path.join(seq_dir, '*.png')))
            mode = random.choice(['1x', '2x', 'large_blur'])

            if mode == '1x' or len(frame_paths) < 50:
                base_rate = random.choice([1, 2])
                blur_img, seq_frames, inp_int, out_int = generate_1x_sequence(
                    frame_paths, window_max=16, output_len=17, base_rate=base_rate
                )
            elif mode == '2x':
                base_rate = random.choice([1, 2])
                blur_img, seq_frames, inp_int, out_int = generate_2x_sequence(
                    frame_paths, window_max=16, output_len=17, base_rate=base_rate
                )
            else:
                max_base = min((len(frame_paths) - 1) // 17, 3)
                base_rate = random.randint(1, max_base)
                blur_img, seq_frames, inp_int, out_int = generate_large_blur_sequence(
                    frame_paths, window_max=16, output_len=17, base_rate=base_rate
                )
            file_name = frame_paths[0]
            num_frames = 16

        # Common conversion and packaging
        # blur_img is a single frame; wrap in batch dim
        blur_input = self.load_frames(np.expand_dims(blur_img, 0))
        # seq_frames is list of frames; stack along time dim
        video = self.load_frames(np.stack(seq_frames, axis=0))

        
        relative_file_name = os.path.relpath(file_name, self.data_dir)
        
        if self.split == 'test':
            # Get base directory and frame prefix
            base_dir = os.path.dirname(relative_file_name)
            frame_stem = os.path.splitext(os.path.basename(relative_file_name))[0]  # "frame_00000"

            # Build new filename
            new_filename = (
                f"{frame_stem}_"
                f"in{in_start:04d}_ie{in_end:04d}_"
                f"os{out_start:04d}_oe{out_end:04d}_"
                f"ctr{center:04d}_win{window:04d}_"
                f"fps{fps:04d}_{mode}.png"
            )

            # Final path
            relative_file_name = os.path.join(base_dir, new_filename)
        
        data = {
            'file_name': relative_file_name,
            'blur_img':  blur_input,
            'num_frames': num_frames,
            'video':     video,
            'caption':   "",
            'mode':      mode,
            'input_interval':  inp_int,
            'output_interval': out_int,
        }
        if self.split == 'test':
            high_fps_video = self.load_frames(np.stack(high_fps_video, axis=0))
            data['high_fps_video'] = high_fps_video
        return data


class BAISTDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

      
        test_folders =  {
            "gWA_sBM_c01_d26_mWA0_ch06_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch04_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch05_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch08_cropped_32X": None,
            "gWA_sBM_c01_d26_mWA0_ch02_cropped_32X": None,
            "gJS_sBM_c01_d02_mJS0_ch08_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch07_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch06_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch03_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch05_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch02_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch03_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch10_cropped_32X": None,
            "gWA_sBM_c01_d26_mWA0_ch10_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch06_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch08_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch06_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch10_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch02_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch04_cropped_32X": None,
            "gPO_sBM_c01_d10_mPO0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch07_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch03_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch04_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch02_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch05_cropped_32X": None,
            "gPO_sBM_c01_d10_mPO0_ch10_cropped_32X": None,
        }

        def collect_blur_images(root_dir, allowed_folders, skip_start=40, skip_end=40):
            blur_image_paths = []

            for dirpath, dirnames, filenames in os.walk(root_dir):
                if os.path.basename(dirpath) == "blur":
                    parent_folder = os.path.basename(os.path.dirname(dirpath))
                    if (self.split in ["test", "val"] and parent_folder in test_folders) or (self.split in "train" and parent_folder not in test_folders):
                        # Filter and sort valid image filenames
                        valid_files = [
                            f for f in filenames
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and os.path.splitext(f)[0].isdigit()
                        ]
                        valid_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

                        # Skip first and last N files
                        middle_files = valid_files[skip_start:len(valid_files) - skip_end]

                        for f in middle_files:
                            from pathlib import Path
                            full_path = Path(os.path.join(dirpath, f))
                            full_output_path = Path("/datasets/sai/gencam/cogvideox/training/cogvideox-baist-test-crop-landscape/deblurred", *full_path.parts[-3:]).with_suffix(".mp4")
                            if not os.path.exists(full_output_path):
                                blur_image_paths.append(os.path.join(dirpath, f))

            return blur_image_paths

    
        self.image_paths = collect_blur_images(self.data_dir, test_folders)
        #if bbx path does not exist, remove the image path
        self.image_paths = [path for path in self.image_paths if os.path.exists(path.replace("blur", "blur_anno").replace(".png", ".pkl"))]

        filtered_image_paths = []
        for blur_path in self.image_paths:
            base_dir = blur_path.replace('/blur/', '/sharp/').replace('.png', '')
            sharp_paths = [f"{base_dir}_{i:03d}.png" for i in range(7)]
            if all(os.path.exists(p) for p in sharp_paths):
                filtered_image_paths.append(blur_path)
        self.image_paths = filtered_image_paths

        if self.split == 'val':
            # Optional: limit validation subset
            self.image_paths = self.image_paths[:4]
        self.length = len(self.image_paths)

    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        blur_img_original = load_as_srgb(image_path)
        def get_optimal_crop(bbx, image_w, image_h):
            bbx = np.array(bbx, dtype=np.float32)
            aspect_ratio = 16 / 9.0

            bbx_w = bbx[2] - bbx[0]
            bbx_h = bbx[3] - bbx[1]
            center_x = (bbx[0] + bbx[2]) / 2.0
            center_y = (bbx[1] + bbx[3]) / 2.0

            # First, try ideal 16:9 crop that just contains the bbox
            crop_w = max(bbx_w, aspect_ratio * bbx_h)
            crop_h = crop_w / aspect_ratio

            # Position the crop centered on the bbox
            crop_x_min = center_x - crop_w / 2.0
            crop_y_min = center_y - crop_h / 2.0
            crop_x_max = center_x + crop_w / 2.0
            crop_y_max = center_y + crop_h / 2.0

            # If the 16:9 crop fits within image boundaries, return it
            if (crop_x_min >= 0 and crop_y_min >= 0 and
                crop_x_max <= image_w and crop_y_max <= image_h):
                return np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max], dtype=np.float32).astype(np.uint32)

            # Otherwise, reduce crop height to fit inside image while still containing bbox
            # Keep width fixed (so bbox fits), reduce height until it fits vertically
            crop_h = min(crop_h, 2.0 * min(center_y, image_h - center_y))
            crop_y_min = center_y - crop_h / 2.0
            crop_y_max = center_y + crop_h / 2

            # Recalculate crop box
            crop_box = np.array([center_x - crop_w / 2.0, crop_y_min,
                                center_x + crop_w / 2.0, crop_y_max], dtype=np.float32)

            # Clamp horizontally if necessary
            if crop_box[0] < 0:
                shift = -crop_box[0]
                crop_box[0] = 0
                crop_box[2] = min(crop_box[2] + shift, image_w)
            elif crop_box[2] > image_w:
                shift = crop_box[2] - image_w
                crop_box[2] = image_w
                crop_box[0] = max(crop_box[0] - shift, 0)

            # Final check: still must contain bbox
            if not (crop_box[0] <= bbx[0] and crop_box[1] <= bbx[1] and
                    crop_box[2] >= bbx[2] and crop_box[3] >= bbx[3]):
                return bbx.astype(np.uint32)  # fallback if impossible

            return crop_box.astype(np.uint32)

        bbx_path = image_path.replace("blur", "blur_anno").replace(".png", ".pkl")
        
        #load the bbx path
        bbx = np.load(bbx_path, allow_pickle=True)['bbox'][0:4]
        # Final crop box
        crop_box = get_optimal_crop(bbx, image_w=blur_img_original.size[0], image_h=blur_img_original.size[1]) #np.array([crop_x_min, crop_y_min, crop_x_max, crop_y_max], dtype=np.float32).astype(np.uint32)
        adjusted_bbx = bbx - np.array([crop_box[0], crop_box[1], crop_box[0], crop_box[1]], dtype=np.float32)
        #turn crop_box into tupel
        print("Crop box: ", crop_box, "File name: ", image_path, "Blur image size: ", blur_img_original.size)
        
        cropped_original = blur_img_original.crop(tuple(crop_box))
        W,H = cropped_original.size
        #save the cropped image
        adjusted_bbx = adjusted_bbx.astype(np.uint32)
        croped_with_adjusted_bbx = cropped_original.crop((adjusted_bbx[0], adjusted_bbx[1], adjusted_bbx[2], adjusted_bbx[3]))
        #save the cropped image
        croped_with_adjusted_bbx.save("/datasets/sai/gencam/cogvideox/training/baist-cropped-adjusted.png")
        cropped_original.save("/datasets/sai/gencam/cogvideox/training/baist-cropped.png")
        blur_img = cropped_original.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)

 #cause pil is width, height
        blur_np = np.array([blur_img])




        base_dir = os.path.dirname(os.path.dirname(image_path))  # strip /blur
        filename = os.path.splitext(os.path.basename(image_path))[0]  # '00000000'
        sharp_dir = os.path.join(base_dir, "sharp")

        frame_paths = [
            os.path.join(sharp_dir, f"{filename}_{i:03d}.png")
            for i in range(7)
        ]


        _, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
                        frame_paths=frame_paths, window_max=7, in_start=0, in_end=7, out_start=0,out_end=7, center=3, mode="1x", fps=240
                    )
        
        
        #crop seq frames using adjusted bbx
        for i in range(len(seq_frames)):
            x1, y1, x2, y2 = map(int, crop_box)
            # print("Crop box: ", crop_box, "File name: ", image_path)
            # print("Frame shape: ", seq_frames[i].shape)
            # Crop using numpy slicing: [y1:y2, x1:x2, :]
            cropped = seq_frames[i][y1:y2, x1:x2]

            # Resize using OpenCV (cv2.resize expects width x height)
            # print("Cropped shape: ", cropped.shape)
            # print("Resizing to: ", (self.image_size[1], self.image_size[0]))
            resized = cv2.resize(cropped, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

            seq_frames[i] = resized
            
        pixel_values = self.load_frames(np.stack(seq_frames, axis=0))
        blur_pixel_values = self.load_frames(blur_np)

        relative_file_name = os.path.relpath(image_path, self.data_dir)

        out_bbx = adjusted_bbx.copy()

        scale_x = blur_pixel_values.shape[3]/W
        scale_y = blur_pixel_values.shape[2]/H
        #scale the bbx
        out_bbx[0] = int(out_bbx[0] * scale_x)
        out_bbx[1] = int(out_bbx[1] * scale_y)
        out_bbx[2] = int(out_bbx[2] * scale_x)
        out_bbx[3] = int(out_bbx[3] * scale_y)

        out_bbx = torch.tensor(out_bbx, dtype=torch.uint32)

        #crop image using the bbx
        blur_img_npy = np.array(blur_img)
        out_bbx_npy = out_bbx.numpy().astype(np.uint32)
        blur_img_npy = blur_img_npy[out_bbx_npy[1]:out_bbx_npy[3], out_bbx_npy[0]:out_bbx_npy[2], :]
        #save the cropped image
        Image.fromarray(blur_img_npy).save("/datasets/sai/gencam/cogvideox/training/baist-cropped-two.png")



    
        data = {
            'file_name': relative_file_name,
            'blur_img': blur_pixel_values,
            'video': pixel_values,
            'bbx': out_bbx,
            'caption': "",
            'input_interval': inp_int,
            'output_interval': out_int,
            "num_frames": num_frames,
            'mode':  "1x",
        }
        return data
    


class GoPro2xMotionBlurDataset(BaseClass):
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

        def get_sharp_paths(blur_paths):
            sharp_paths = []
            for blur_path in blur_paths:
                base_dir = blur_path.replace('/blur/', '/sharp/')
                frame_num = int(os.path.basename(blur_path).split('.')[0])
                dir_path = os.path.dirname(base_dir)
                sequence = [
                    os.path.join(dir_path, f"{frame_num + offset:06d}.png")
                    for offset in range(-6, 7)
                ]
                if all(os.path.exists(path) for path in sequence):
                    sharp_paths.append(sequence)
            return sharp_paths




        self.blur_paths = sorted(glob.glob(pattern))
        filtered_blur_paths = []
        for path in self.blur_paths:
            full_output_path = Path("/datasets/sai/gencam/cogvideox/training/cogvideox-gopro-2x/deblurred", *path.split('/')[-2:]).with_suffix(".mp4")
            if not os.path.exists(full_output_path):
                filtered_blur_paths.append(path)
        self.blur_paths = filtered_blur_paths

        self.sharp_paths = get_sharp_paths(self.blur_paths)
        if self.split == 'val':
            # Optional: limit validation subset
            self.sharp_paths = self.sharp_paths[:5]
        self.length = len(self.sharp_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Path to the blurred (center) frame
        sharp_path = self.sharp_paths[idx]


        # Load sharp frames
        blur_img, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
                        frame_paths=sharp_path, window_max=13, in_start=3, in_end=10, out_start=0,out_end=13, center=6, mode="2x", fps=240
                    )
        
        # Convert to pixel values via BaseClass loader
        video = self.load_frames(np.array(seq_frames))                    # shape: (output_length, H, W, C)
        blur_input = self.load_frames(np.expand_dims(np.array(blur_img), 0))  # shape: (1, H, W, C)
        last_two_parts_of_path = os.path.join(*sharp_path[6].split(os.sep)[-2:])
        #print(f"Time taken to load and process data: {end_time - start_time:.2f} seconds")
        data = {
            'file_name': last_two_parts_of_path,
            'blur_img': blur_input,
            'video': video,
            "caption": "",
            'input_interval': inp_int,
            'output_interval': out_int,
            "num_frames": num_frames,
            "mode": "2x",
        }
        return data
    

class GoProLargeMotionBlurDataset(BaseClass):
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

        def get_sharp_paths(blur_paths):
            sharp_paths = []
            for blur_path in blur_paths:
                base_dir = blur_path.replace('/blur/', '/sharp/')
                frame_num = int(os.path.basename(blur_path).split('.')[0])
                dir_path = os.path.dirname(base_dir)
                sequence = [
                    os.path.join(dir_path, f"{frame_num + offset:06d}.png")
                    for offset in range(-16, 16)
                ]
                if all(os.path.exists(path) for path in sequence):
                    sharp_paths.append(sequence)
            return sharp_paths

        self.blur_paths = sorted(glob.glob(pattern))
        self.sharp_paths = get_sharp_paths(self.blur_paths)
        if self.split == 'val':
            # Optional: limit validation subset
            self.sharp_paths = self.sharp_paths[:5]
        self.length = len(self.sharp_paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Path to the blurred (center) frame
        sharp_path = self.sharp_paths[idx]


        # Load sharp frames
        blur_img, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
                        frame_paths=sharp_path, window_max=32, in_start=0, in_end=32, out_start=0,out_end=32, center=16, mode="lb", fps=120
                    )
        
        # Convert to pixel values via BaseClass loader
        video = self.load_frames(np.array(seq_frames))                    # shape: (output_length, H, W, C)
        blur_input = self.load_frames(np.expand_dims(np.array(blur_img), 0))  # shape: (1, H, W, C)
        last_two_parts_of_path = os.path.join(*sharp_path[6].split(os.sep)[-2:])
        #print(f"Time taken to load and process data: {end_time - start_time:.2f} seconds")
        data = {
            'file_name': last_two_parts_of_path,
            'blur_img': blur_input,
            'video': video,
            "caption": "",
            'input_interval': inp_int,
            'output_interval': out_int,
            "num_frames": num_frames,
            "mode": "lb",
        }
        if self.split == 'test':
            high_fps_video = self.load_frames(np.stack(high_fps_video, axis=0))
            data['high_fps_video'] = high_fps_video
        return data
    
class SimpleBAISTDataset(BaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

      
        test_folders =  {
            "gWA_sBM_c01_d26_mWA0_ch06_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch04_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch05_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch08_cropped_32X": None,
            "gWA_sBM_c01_d26_mWA0_ch02_cropped_32X": None,
            "gJS_sBM_c01_d02_mJS0_ch08_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch07_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch06_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch03_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch05_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch02_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch03_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch10_cropped_32X": None,
            "gWA_sBM_c01_d26_mWA0_ch10_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch06_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch08_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch06_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch10_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch02_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch04_cropped_32X": None,
            "gPO_sBM_c01_d10_mPO0_ch09_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch07_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch03_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch04_cropped_32X": None,
            "gBR_sBM_c01_d05_mBR0_ch02_cropped_32X": None,
            "gHO_sBM_c01_d20_mHO0_ch01_cropped_32X": None,
            "gMH_sBM_c01_d22_mMH0_ch05_cropped_32X": None,
            "gPO_sBM_c01_d10_mPO0_ch10_cropped_32X": None,
        }

        def collect_blur_images(root_dir, allowed_folders, skip_start=40, skip_end=40):
            blur_image_paths = []

            for dirpath, dirnames, filenames in os.walk(root_dir):
                if os.path.basename(dirpath) == "blur":
                    parent_folder = os.path.basename(os.path.dirname(dirpath))
                    if (self.split in ["test", "val"] and parent_folder in test_folders) or (self.split in "train" and parent_folder not in test_folders):
                        # Filter and sort valid image filenames
                        valid_files = [
                            f for f in filenames
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) and os.path.splitext(f)[0].isdigit()
                        ]
                        valid_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

                        # Skip first and last N files
                        middle_files = valid_files[skip_start:len(valid_files) - skip_end]

                        for f in middle_files:
                            from pathlib import Path
                            full_path = Path(os.path.join(dirpath, f))
                            full_output_path = Path("/datasets/sai/gencam/cogvideox/training/cogvideox-simplebaist-finetune-sample/deblurred", *full_path.parts[-3:]).with_suffix(".mp4")
                            if not os.path.exists(full_output_path) or self.split in ["train", "val"]:
                                blur_image_paths.append(os.path.join(dirpath, f))

            return blur_image_paths

    

        self.image_paths = collect_blur_images(self.data_dir, test_folders)
        print("Blur image paths: ", len(self.image_paths))
        #if bbx path does not exist, remove the image path
        self.image_paths = [path for path in self.image_paths if os.path.exists(path.replace("blur", "blur_anno").replace(".png", ".pkl"))]

        filtered_image_paths = []
        for blur_path in self.image_paths:
            base_dir = blur_path.replace('/blur/', '/sharp/').replace('.png', '')
            sharp_paths = [f"{base_dir}_{i:03d}.png" for i in range(7)]
            if all(os.path.exists(p) for p in sharp_paths):
                filtered_image_paths.append(blur_path)

        self.image_paths = filtered_image_paths
                
        if self.split == 'val':
            # Optional: limit validation subset
            self.image_paths = self.image_paths[:4]
        self.length = len(self.image_paths)

    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        blur_img_original = load_as_srgb(image_path)

        bbx_path = image_path.replace("blur", "blur_anno").replace(".png", ".pkl")
        
        #load the bbx path
        bbx = np.load(bbx_path, allow_pickle=True)['bbox'][0:4]
        # Final crop box
        #turn crop_box into tupel
        W,H = blur_img_original.size
        blur_img = blur_img_original.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)

 #cause pil is width, height
        blur_np = np.array([blur_img])




        base_dir = os.path.dirname(os.path.dirname(image_path))  # strip /blur
        filename = os.path.splitext(os.path.basename(image_path))[0]  # '00000000'
        sharp_dir = os.path.join(base_dir, "sharp")

        frame_paths = [
            os.path.join(sharp_dir, f"{filename}_{i:03d}.png")
            for i in range(7)
        ]


        _, seq_frames, inp_int, out_int, high_fps_video, num_frames = generate_test_case(
                        frame_paths=frame_paths, window_max=7, in_start=0, in_end=7, out_start=0,out_end=7, center=3, mode="1x", fps=240
                    )
        

            
        pixel_values = self.load_frames(np.stack(seq_frames, axis=0))
        blur_pixel_values = self.load_frames(blur_np)

        relative_file_name = os.path.relpath(image_path, self.data_dir)

        out_bbx = bbx.copy()

        scale_x = blur_pixel_values.shape[3]/W
        scale_y = blur_pixel_values.shape[2]/H
        #scale the bbx
        out_bbx[0] = int(out_bbx[0] * scale_x)
        out_bbx[1] = int(out_bbx[1] * scale_y)
        out_bbx[2] = int(out_bbx[2] * scale_x)
        out_bbx[3] = int(out_bbx[3] * scale_y)

        out_bbx = torch.tensor(out_bbx, dtype=torch.uint32)

        #crop image using the bbx
        blur_img_npy = np.array(blur_img)
        out_bbx_npy = out_bbx.numpy().astype(np.uint32)
        blur_img_npy = blur_img_npy[out_bbx_npy[1]:out_bbx_npy[3], out_bbx_npy[0]:out_bbx_npy[2], :]
        #save the cropped image
        Image.fromarray(blur_img_npy).save("/datasets/sai/gencam/cogvideox/training/baist-cropped-two.png")



    
        data = {
            'file_name': relative_file_name,
            'blur_img': blur_pixel_values,
            'video': pixel_values,
            'bbx': out_bbx,
            'caption': "",
            'input_interval': inp_int,
            'output_interval': out_int,
            "num_frames": num_frames,
            'mode':  "1x",
        }
        return data
    