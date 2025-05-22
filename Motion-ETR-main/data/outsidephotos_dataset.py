import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class OutsidePhotosDataset(BaseDataset):
    def __init__(self, opt):
        # super(AlignedDataset,self).__init__(opt)
        self.opt = opt
        self.root = opt.dataroot
        # self.dir_AB = "/home/tedlasai/genCamera/BlurryOutsidePhotos"
        # self.AB_paths = sorted(make_dataset(self.dir_AB))

        self.dir_AB = "/home/tedlasai/genCamera/BlurryOutsidePhotos"
        self.AB_paths = sorted(make_dataset(self.dir_AB))

        print("len(self.AB_paths) = ", len(self.AB_paths))
        


        # we use more data augmentation
        # if opt.phase == 'train':
        #     multiframe_data = ['9frame']
        #     dir_root = os.path.dirname(self.root)
        #     for i in multiframe_data:
        #         dir_multiframe = os.path.join(dir_root,"gopro_%s"%i,opt.phase)
        #         multiframe_paths = sorted(make_dataset(dir_multiframe))
        #         self.AB_paths += multiframe_paths
        #assert(opt.resize_or_crop == 'resize_and_crop')

        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5, 0.5, 0.5),
        #                                        (0.5, 0.5, 0.5))]   
        if opt.phase == "train":
            transform_list = [#transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                            transforms.ToTensor()]
        elif opt.phase == 'test':
            transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        print(AB.size)

        # data aug with resize
        # input_sizeX,input_sizeY = AB.size[0],AB.size[1] 
        # if self.opt.phase =='train':
        #     # if random.random() < 0.5:
        #     resize = random.uniform(1,1.5) 
        #     AB = AB.resize((int(input_sizeX * resize), int(input_sizeY*resize)), Image.BICUBIC)
            
        AB = self.transform(AB)
        AB = AB.permute(0,1,2)

        A = AB.clone()
        B = AB.clone()

        print("AB.size() = ", AB.size())
        # # if not crop in test time
        # if self.opt.no_crop:
        #     A = AB[:,:,:w]
        #     B = AB[:,:,w:w_total]
        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)
        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(1, idx)
        #     B = B.index_select(1, idx)

        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'OutsidePhotosDataset'