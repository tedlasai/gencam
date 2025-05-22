import time
import os

import ipdb
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from util.metrics import PSNR
#from ssim import SSIM
from PIL import Image
from contextlib import contextmanager
import scipy.io as sio
import numpy as np 
import cv2

@contextmanager
def timer(name):
	t1 = time.time()
	yield 
	print(name, ":   ", time.time() - t1)

def write_txt(file_name, line):
	with open(file_name,'a') as log:
		log.write(line+'\n')
	print(line)



def EPE(mat, offset):

	S_point = np.squeeze(offset[:,0:2,:,:])
	E_point = np.squeeze(offset[:,-2:,:,:])
	vec = S_point - E_point
	vec = np.transpose(vec,(1,2,0))
	# index = np.where(vec[...,1] < 0)
	# vec[index] = -vec[index]
	vec = np.flip(vec,axis=-1)
	
	crop_edge = 30
	vec_crop = vec[crop_edge:-crop_edge,crop_edge:-crop_edge]
	mat_crop = mat[crop_edge:-crop_edge,crop_edge:-crop_edge]
	vec_crop = np.abs(vec_crop)
	mat_crop = np.abs(mat_crop)
	error = np.mean(np.square( vec_crop - mat_crop))

	return error

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.dataroot="/home/tedlasai/genCamera/Motion-ETR-main/Gopro_align_data"

print("Building dataloader")
data_loader = CreateDataLoader(opt)
print("Building dataset")
dataset = data_loader.load_data()
print("Building model")
model = create_model(opt)
print("Finish building model")
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
Record_txt =  os.path.join(opt.results_dir, opt.name,'EPE_record.txt')
# test
avgPSNR = 0.0
avgSSIM = 0.0
avgEPE = 0.0
avgEPE_new = 0.0
calculate_EPE = False
counter = 0

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
	new_size = (W // downsample_spatial, H // downsample_spatial)
	
	# PNG compression param: 0 = no compression
	png_params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
	
	for idx, frame in enumerate(frames):
		# frame is RGB; convert to BGR for OpenCV
		bgr = frame[..., ::-1]
		if downsample_spatial > 1:
			bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_NEAREST)
		
		filename = os.path.join(output_dir, f"frame_{idx:05d}.png")
		success = cv2.imwrite(filename, bgr, png_params)
		if not success:
			raise RuntimeError(f"Failed to write frame {idx} to {filename}")
		

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	counter = i+1
	# import ipdb; ipdb.set_trace()
	paths = model.set_input(data)
	with timer('test time'):
		model.test()
		frames = model.vis_everyframe(target_n_offset=16)
		frames = np.array(frames)
		base_name = os.path.basename(paths[0])   
		#remove png extension
		base_folder_name = os.path.splitext(base_name)[0] + "+_1x"
		print("Frames shape: ", frames.shape)

		save_frames_as_pngs(os.path.join(opt.results_dir, opt.name, 'videos', "outsidephotos", base_folder_name), frames, downsample_spatial=1)
	visuals = model.get_current_visuals()
	if calculate_EPE:
		offset = model.get_current_offset()
		path = model.get_image_paths()
		dir_name, img_name = os.path.split(path[0])
		img_id = os.path.splitext(img_name)[0]
		mat_name = img_id + '_mfmap.mat'
		mat = sio.loadmat(os.path.join(dir_name, mat_name))['mfmap']
		epe, epe_new = EPE(mat,offset)
		print('EPE image pair %s: %f'%(img_id,epe))
		avgEPE += epe
		avgEPE_new += epe_new

	crop_edge = 0
	if opt.blur_direction == 'reblur':
		psnr = PSNR(visuals['Reblur'],visuals['Blurry'])
		avgPSNR += psnr
		pilFake = Image.fromarray(visuals['Reblur'])
		pilReal = Image.fromarray(visuals['Blurry'])
	# if opt.blur_direction == 'deblur':
	# 	psnr = PSNR(visuals['Restore'],visuals['Sharp'])
	# 	avgPSNR += psnr
	# 	pilFake = Image.fromarray(visuals['Restore'])
	# 	pilReal = Image.fromarray(visuals['Sharp'])
	
	#avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
	img_path = model.get_image_paths()
	#print('process image %s \t psnr:%.4f' % (img_path,psnr))
	visualizer.save_images(webpage, visuals, img_path)


avgPSNR /= counter
#avgSSIM /= counter
avgEPE /= counter
line ='PSNR = %f' %(avgPSNR)
write_txt(Record_txt,line)
line = 'aveEPE= %f'%(avgEPE)
write_txt(Record_txt,line)
line = 'aveEPE_new= %f'%(avgEPE_new/counter)
write_txt(Record_txt,line)
# webpage.save()
