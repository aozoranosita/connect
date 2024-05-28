from torch.utils.data import DataLoader, Dataset
import imageio
import os, sys
import numpy as np
import pandas as pd
import torch
from affinity import compute_affinity_maps_3d
from volumentations import *

NUM_CLASS = 7
PATCH_SIZE = (64,64,64)

def get_augmentation(patch_size):
    return Compose([
        Rotate((-15,15),(-15,15),(-15,15)),
        RandomCropFromBorders(crop_value=0.1, p=0.5),
        ElasticTransform((0, 0.25), interpolation=2, p=0.1),#time consuming
        Resize(patch_size,interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Normalize(always_apply=True),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
        GaussianNoise(var_limit=(0, 5), p=0.2),
        #RandomGamma(gamma_limit=(80, 120), p=0.2),
    ], p=1.0)

class ConnectomicsDataset(Dataset):
    def __init__(self, root, csvfile_path, phase='train', transform=get_augmentation(PATCH_SIZE)):
        self.root = root
        self.phase = phase
        self.transform = transform
        csvfile_path = os.path.join(root, csvfile_path)
        self.df = self.mk_df(phase, csvfile_path)

    def mk_df(self,phase,csvfile_path):
        origin = pd.read_csv(csvfile_path)
        img_path = origin.columns[1]
        label_path = origin.columns[2]
        self.img_dir = os.path.join(self.root,img_path)
        self.label_dir = os.path.join(self.root,label_path)
        
        if phase == "train":
            df = origin[origin["fold"]!=2]
        elif phase == "val":
            df = origin[origin["fold"]==2]
        elif phase == "test":
            df = origin
        else:
            sys.exit('phase should "train" or "val" or "test"')
        return df.reset_index(drop=True) #__getitem__ -> call data using index, so dataflame should 0,1,2,,

    def __len__(self):
        return self.df.shape[0]
    
    def get_df(self):
        return self.df

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir,self.df.at[idx,'img'])
        label_path = os.path.join(self.label_dir,self.df.at[idx,'label'])
        image = imageio.volread(img_path)
        label = imageio.volread(label_path)
        if self.transform:
            data = {'image': image, 'mask': label}
            aug = self.transform
            aug_data = aug(**data)
            image, label = aug_data['image'], aug_data['mask']
        
        affinity_maps = compute_affinity_maps_3d(label)
        image = torch.tensor(image, dtype=torch.float16).unsqueeze(0)
        label = torch.tensor(affinity_maps, dtype=torch.float16)
        return image, label


if __name__=='__main__':
    val_dataset = ConnectomicsDataset("data/", "fold.csv", phase="val")

    #dataloader
    batch_size = 1
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    im_ex,label_ex = next(iter(val_dataloader))
    #print(im_ex[0])
    print(label_ex[0,0,1,1,3:10])
    print(label_ex.shape)
    print(im_ex.shape)
    #torch.Size([1, 3, 64, 64, 64])
    #torch.Size([1, 1, 64, 64, 64])
