from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import utils
from model import centerEsti
from model import F26_N9
from model import F17_N9
from model import F35_N8
import torch.nn.functional as F


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Training settings
parser = argparse.ArgumentParser(description='parser for video prediction')
parser.add_argument('--input', type=str, required=True, help='input image')
parser.add_argument('--cuda', action='store_true', help='use cuda')

args = parser.parse_args()
# load model
print("Built Model 1")
model1 = centerEsti()
print("Built Model 2")
model2 = F35_N8()
print("Built Model 3")
model3 = F26_N9()
print("Built Model 4")
model4 = F17_N9()
print("Loading model weights")
checkpoint = torch.load('models/center_v3.pth')
model1.load_state_dict(checkpoint['state_dict_G'])
print("Loading model weights 2")
checkpoint = torch.load('models/F35_N8.pth', weights_only=False)
model2.load_state_dict(checkpoint['state_dict_G'])
print("Loading model weights 3")
checkpoint = torch.load('models/F26_N9_from_F35_N8.pth', weights_only=False)
model3.load_state_dict(checkpoint['state_dict_G'])
print("Loading model weights 4")
checkpoint = torch.load('models/F17_N9_from_F26_N9_from_F35_N8.pth', weights_only=False)
model4.load_state_dict(checkpoint['state_dict_G'])

if args.cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()

model1.eval()
model2.eval()
model3.eval()
model4.eval()

inputFile = args.input
input = utils.load_image(inputFile)
width, height= input.size
input = input.crop((0,0, width-width%20, height-height%20))
input_transform = transforms.Compose([
    transforms.ToTensor(),
])
input = input_transform(input)
print("input shape", input.shape)
input = input.unsqueeze(0)
#downsample input by 45%
# downsample to 45% of original size
scale_factor = 0.5
input_ds = F.interpolate(
    input,
    scale_factor=scale_factor,
    mode='bilinear',       # or 'bicubic' for smoother high-res downsampling
    align_corners=False
)
input = input_ds
# input = input [:,:3,:600,:600]
# print(input)
#print("downsampled shape", input_ds.shape)

if args.cuda:
    input = input.cuda()
#input = Variable(input, volatile=True)


output4 = model1(input)
output3_5 = model2(input, output4)
output2_6 = model3(input, output3_5[0], output4, output3_5[1])
output1_7 = model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])

if args.cuda:
    output1 = output1_7[0].cpu()
    output2 = output2_6[0].cpu()
    output3 = output3_5[0].cpu()
    output4 = output4.cpu()
    output5 = output3_5[1].cpu()
    output6 = output2_6[1].cpu()
    output7 = output1_7[1].cpu()
else:
    output1 = output1_7[0]
    output2 = output2_6[0]
    output3 = output3_5[0]
    output4 = output4
    output5 = output3_5[1]
    output6 = output2_6[1]
    output7 = output1_7[1]
output_data = output1.data[0] * 255
utils.save_image(inputFile[:-4] + '-esti1' + inputFile[-4:], output_data)                
output_data = output2.data[0]*255
utils.save_image(inputFile[:-4] + '-esti2' + inputFile[-4:], output_data)                
output_data = output3.data[0]*255
utils.save_image(inputFile[:-4] + '-esti3' + inputFile[-4:], output_data)
output_data = output4.data[0]*255
utils.save_image(inputFile[:-4] + '-esti4' + inputFile[-4:], output_data)
output_data = output5.data[0]*255
utils.save_image(inputFile[:-4] + '-esti5' + inputFile[-4:], output_data)
output_data = output6.data[0]*255
utils.save_image(inputFile[:-4] + '-esti6' + inputFile[-4:], output_data)
output_data = output7.data[0]*255
utils.save_image(inputFile[:-4] + '-esti7' + inputFile[-4:], output_data)
