import torch
import torch.utils.data
import PIL
import os
import numpy as np
from glob import glob
import random

def My_loader(path):
    return PIL.Image.open(path).convert('RGB')

class keypoint_customData(torch.utils.data.Dataset):
    def __init__(self, datapath='.', data_transforms=None, loader=My_loader):
        self.kid_files = glob('/data2/zhousiyu/dataset/face_data/frame_and_keypoint/kid_img/*.jpg')
        self.old_files = glob('/data2/zhousiyu/dataset/face_data/frame_and_keypoint/old_img/*.jpg')
        self.loader = loader
        self.data_transforms = data_transforms
        kid_len = len(self.kid_files)
        old_len = len(self.old_files)
        if kid_len < old_len:
            self.length = kid_len
        else:
            self.length = old_len

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        A_index = random.randint(0, self.length-1)
        B_index = random.randint(0, self.length-1)

        A = self.loader(self.kid_files[A_index])
        B = self.loader(self.old_files[B_index])
        data_A = A.crop((256, 0, 512, 256))
        data_B = B.crop((256, 0, 512, 256))

        if self.data_transforms is not None:
            data_A = self.data_transforms(data_A)
            data_B = self.data_transforms(data_B)


        
        return {'A': data_A, 'B': data_B, 'A_paths': self.kid_files[A_index], 'B_paths': self.old_files[B_index]}