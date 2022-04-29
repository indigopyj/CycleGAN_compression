import os
import pickle
from idna import valid_contextj

import numpy as np
from PIL import Image
from torchvision import transforms

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

import torch


class VideoDataset(BaseDataset):
    def __init__(self, opt):
        super(VideoDataset, self).__init__(opt)
        BaseDataset.__init__(self, opt)
        if opt.Viper2Cityscapes:
            if opt.phase == "test" or opt.phase == "val":
                phase = "val"
            else:
                phase = "train"
            self.dir_A = os.path.join(opt.dataroot, 'Viper', phase, 'img')
            self.dir_B = os.path.join(opt.dataroot, "Cityscapes_sequence", "leftImg8bit_sequence", opt.phase)
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')  # create a path '/path/to/data/trainB'
        
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.seq_list = sorted(os.listdir(self.dir_A))
        input_nc = self.opt.input_nc
        self.transform = get_transform(self.opt, grayscale=(input_nc == 1))
        # self.transform = transforms.Compose([
        #     transforms.Resize((opt.load_size, opt.load_size), Image.BICUBIC),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])

    def __getitem__(self, index):
        if self.opt.Viper2Cityscapes:
            seq_path = os.path.join(self.dir_A, self.seq_list[index])
            A_path = sorted(os.listdir(seq_path))
            interval = torch.randint(1, self.opt.max_interval, [1]).item()
            if self.opt.phase != 'train':
                idx1 = torch.randint(0, len(A_path) - self.opt.max_interval, [1]).item()
            else:
                idx1 = torch.randint(0, len(A_path) - interval, [1]).item()
            img_root = seq_path
        else: # ffs dataset
            A_path = sorted(os.listdir(self.dir_A))
            interval = torch.randint(1, self.opt.max_interval, [1]).item()
            idx1 = index
            img_root = self.dir_A
            if idx1 >= len(A_path) - interval: # clipping
                idx1 = len(A_path) - interval - 1
        if self.opt.phase == "test":
            return {'seq_path' : seq_path }
        img1 = Image.open(os.path.join(img_root, A_path[idx1])).convert("RGB") # change
        img2 = Image.open(os.path.join(img_root, A_path[idx1 + interval])).convert("RGB") #change

        # FIXME: crop 위치가 random이 아닌 것 같습니다
        # 두 이미지에 같은 random crop 적용
        # if self.opt.isTrain and np.random.rand(1)[0] < self.opt.crop_prob:
        #     h = img1.size[0]  # 모든 이미지는 이미 정사각형
        #     crop_scale_x = np.random.uniform(self.opt.crop_scale, 1)
        #     crop_scale_y = np.random.uniform(self.opt.crop_scale, 1)
        #     x = int(h * (1 - crop_scale_x))  # random crop 시작 좌표 구하기
        #     y = int(h * (1 - crop_scale_y))  # random crop 시작 좌표 구하기
        #     crop_size = int(h * min(crop_scale_x, crop_scale_y))
        #     img1 = img1.crop((x, y, x + crop_size, y + crop_size))
        #     img2 = img2.crop((x, y, x + crop_size, y + crop_size))

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        

        return {'img1': img1, 'img2': img2, "img1_paths": A_path[idx1], "img_root": img_root}

        
        

    def __len__(self):
        return len(self.seq_list)  # train: 757