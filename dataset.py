from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os, sys
import numpy as np
import pandas as pd
import torch
from affinity import compute_affinity_maps_3d
from volumentations import *

NUM_CLASS = 7

class ConnectomicsDataset(Dataset):
    def __init__(self, root, csvfile_path, phase='train', transform=None):
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
        image = Image.open(img_path)
        label = Image.open(label_path)
        e = np.eye(NUM_CLASS,dtype = bool)
        label = e[label]
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

def get_augmentation(patch_size):
    return Compose([
        Resize(patch_size, always_apply=True),
        Normalize(always_apply=True),
        ElasticTransform((0, 0.25)),
        Rotate((-15,15),(-15,15),(-15,15)),
        RandomGamma(),
        GaussianNoise(),
    ], p=0.8)


if __name__=='__main__':
    val_dataset = ConnectomicsDataset("data/", "fold.csv", phase="val", transform=ToTensor())

    #dataloader
    batch_size = 1
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    im_ex,label_ex = next(iter(val_dataloader))
    #print(im_ex[0])
    print(label_ex[0])
    print(label_ex[0].shape)

    data = {'image': im_ex, 'mask': label_ex}
    # Augmentationを適用
    aug = get_augmentation()
    aug_data = aug(**data)
    img_aug, lbl_aug = aug_data['image'], aug_data['mask']
