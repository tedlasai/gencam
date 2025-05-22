import os
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from model import centerEsti, F35_N8, F26_N9, F17_N9
import utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def save_frames_as_pngs(output_dir, video_array,
                        downsample_spatial=1,   # e.g. 2 to halve width & height
                        downsample_temporal=1): # e.g. 2 to keep every 2nd frame
    """
    Save each frame of a (T, H, W, C) numpy array as a PNG with no compression.
    """
    assert video_array.ndim == 4 and video_array.shape[-1] == 3, \
        "Expected (T, H, W, C=3) array"
    assert video_array.dtype == np.uint8, "Expected uint8 array"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # temporal downsample
    frames = video_array[::downsample_temporal]
    
    # compute spatially downsampled size
    T, H, W, _ = frames.shape
    new_size = (int(W / downsample_spatial), int(H / downsample_spatial))
    
    # PNG compression param: 0 = no compression
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    
    for idx, frame in enumerate(frames):
        # frame is RGB; convert to BGR for OpenCV
        bgr = frame[..., ::-1]
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
        
        filename = os.path.join(output_dir, "frame_{:05d}.png".format(idx))
        success = cv2.imwrite(filename, bgr, png_params)
        if not success:
            raise RuntimeError("Failed to write frame ")

def load_models(model_dir, use_cuda=False):
    # instantiate
    m1 = centerEsti()
    m2 = F35_N8()
    m3 = F26_N9()
    m4 = F17_N9()
    # load weights
    ck = torch.load(os.path.join(model_dir, 'center_v3.pth'))
    m1.load_state_dict(ck['state_dict_G'])
    ck = torch.load(os.path.join(model_dir, 'F35_N8.pth'), weights_only=False)
    m2.load_state_dict(ck['state_dict_G'])
    ck = torch.load(os.path.join(model_dir, 'F26_N9_from_F35_N8.pth'), weights_only=False)
    m3.load_state_dict(ck['state_dict_G'])
    ck = torch.load(os.path.join(model_dir, 'F17_N9_from_F26_N9_from_F35_N8.pth'), weights_only=False)
    m4.load_state_dict(ck['state_dict_G'])
    # cuda
    if use_cuda:
        m1, m2, m3, m4 = m1.cuda(), m2.cuda(), m3.cuda(), m4.cuda()
    for m in (m1, m2, m3, m4): m.eval()
    return m1, m2, m3, m4


def process_file(path, models, use_cuda, scale=0.5):
    m1, m2, m3, m4 = models
    img = utils.load_image(path)
    w, h = img.size
    img = img.crop((0, 0, w - w % 20, h - h % 20))
    tensor = transforms.ToTensor()(img).unsqueeze(0)
    tensor = F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
    if use_cuda:
        tensor = tensor.cuda()
    # inference
    o4 = m1(tensor)
    o3, o5 = m2(tensor, o4)
    o2, o6 = m3(tensor, o3, o4, o5)
    o1, o7 = m4(tensor, o2, o3, o5, o6)
    outputs = [o1, o2, o3, o4, o5, o6, o7]
    # collect
    frames = []
    for out in outputs:
        cpu = out[0].detach().cpu() if use_cuda else out[0]
        img_arr = np.clip((cpu.data.numpy() * 255), 0,255).astype(np.uint8)
        # shape CHW to HWC
        img_arr = img_arr.transpose(1, 2, 0)
        frames.append(img_arr)
    return np.stack(frames, axis=0)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--blur_root',  required=True)
    p.add_argument('--model_dir',  required=True)
    p.add_argument('--output_root',required=True)
    p.add_argument('--downsample',  required=False, type=float, default=1)
    p.add_argument('--add_frames',  required=False, type=float, default=0)
    p.add_argument('--cuda',      action='store_true')
    args = p.parse_args()

    models = load_models(args.model_dir, use_cuda=args.cuda)
    for bd, _, _ in os.walk(args.blur_root):
        rel = os.path.relpath(bd, args.blur_root)
        odir = os.path.join(args.output_root, rel)
        image_files = sorted(glob.glob(os.path.join(bd, '*.png')))
        image_files += sorted(glob.glob(os.path.join(bd, '*.jpg')))
        for f in image_files:
            idx = os.path.splitext(os.path.basename(f))[0]
           
            vid = process_file(f, models, args.cuda)
            
            if args.add_frames == 9: #outside mode
                # Step 1: Repeat the middle frame (index 3)
                vid_extend = np.concatenate([vid[:4], vid[3:4], vid[4:]], axis=0)  # Shape: [8, H, W, C]

                # Step 2: Repeat each frame
                vid = np.repeat(vid_extend, repeats=2, axis=0)  # Shape: [16, H, W, C]
                outp = os.path.join(odir, "{}_1x".format(idx))
            else:
                outp = os.path.join(odir, "{}".format(idx))

            save_frames_as_pngs(outp, vid, downsample_spatial=args.downsample)
            print("Saved video {}".format(outp))
            
            

if __name__ == '__main__':
    main()

#using p312
#python generate_favaro_videos.py   --blur_root /home/tedlasai/genCamera/GOPRO_7/test/blur   --model_dir  /home/tedlasai/genCamera/Favaro/models   --output_root /home/tedlasai/genCamera/GOPROResultsImages/Favaro   --cuda
#python generate_favaro_videos.py   --blur_root /home/tedlasai/genCamera/BlurryOutsidePhotos   --model_dir  /home/tedlasai/genCamera/Favaro/models   --output_root /home/tedlasai/genCamera/Favaro/outsideimageresults   --cuda --downsample 0.5 --add_frames 9