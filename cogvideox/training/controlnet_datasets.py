import os
import glob
import random


import cv2
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from decord import VideoReader
from torch.utils.data.dataset import Dataset
from controlnet_aux import CannyDetector, HEDdetector


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
            stride=(1, 2), 
            sample_n_frames=25,
            hflip_p=0.5,
            controlnet_type='canny',
            *args,
            **kwargs
        ):
        self.height, self.width = unpack_mm_params(image_size)
        self.stride_min, self.stride_max = unpack_mm_params(stride)
        self.data_dir = data_dir
        self.sample_n_frames = sample_n_frames
        self.hflip_p = hflip_p
        
        self.length = 0
        
        self.controlnet_processor = init_controlnet(controlnet_type)
        
    def __len__(self):
        return self.length
        
    def load_video_info(self, video_path):
        video_reader = VideoReader(video_path)
        fps_original = video_reader.get_avg_fps()
        video_length = len(video_reader)
        
        sample_stride = random.randint(self.stride_min, self.stride_max)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        start_idx   = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        np_video = video_reader.get_batch(batch_index).asnumpy()
        pixel_values = torch.from_numpy(np_video).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 127.5 - 1
        return pixel_values
    
    def load_frames(self, frames):
        video_length = frames.shape[1]
        sample_stride = random.randint(self.stride_min, self.stride_max)
        clip_length = min(video_length, (self.sample_n_frames - 1) * sample_stride + 1)
        pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 127.5 - 1
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
    def __init__(self, split, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # videos_paths = glob.glob(os.path.join(self.video_root_dir, '*.mp4'))
        # videos_names = set([os.path.basename(x) for x in videos_paths])
        # self.df = pd.read_csv(csv_path)
        # self.df['checked'] = self.df['path'].map(lambda x: int(x in videos_names))
        # self.df = self.df[self.df['checked'] == True]
        # self.data_dir = data_dir
        self.split = split
        
        if self.split == 'train':
            self.data_dir = os.path.join(self.data_dir, 'test_blur')
        
        elif self.split in ['val', 'test']:
            self.data_dir = os.path.join(self.data_dir, 'test_blur')


        self.image_paths = sorted(glob.glob(os.path.join(self.data_dir, '*/*_w09.png')))#only get files with _w9

        if self.split == "val":
            self.image_paths = ["/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0200/00249_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/IMG_0167/00017_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/train_blur/720p_240fps_5/00249_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0015/00169_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/GOPR9637a/00321_w09.png",
                                "/datasets/sai/gencam/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur/IMG_0183/00505_w09.png",
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
        motion_blur_amount = int(filename.split("_")[1][1:3])//2 #get first two letters

        sharp_directory = directory.replace('test_blur', 'full_sharp').replace('train_blur', 'full_sharp')
        sharp_filename = os.path.join(sharp_directory, f"{img_id}.png")

        sharp_image_start = int(img_id) - motion_blur_amount
        sharp_image_end = int(img_id) + motion_blur_amount + 1

        images = []
        for i in range(sharp_image_start, sharp_image_end):
            #convert to 5 digits
            i_str = str(i).zfill(5)
            current_sharp_image = os.path.join(sharp_directory, f"{str(i).zfill(5)}.png")
            img = Image.open(current_sharp_image).convert("RGB")
            #append image 4 times to the list
            images.extend([np.array(img)] * (5 if i == sharp_image_start else 4))
            
        images = np.array(images) #F*H*W*C

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
            'motion_blur_amount': motion_blur_amount
        }
        return data
