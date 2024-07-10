import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset import ConnectomicsDataset
from model import ResidualSymmetricUNet3D
import imageio

MAX_SIZE = (128,128,128)
DATA = "../image/test-input.tif"
WEIGHT  = 'residual_symmetric_unet3d.pth'
SAVE_PATH = "../pred/test.npy"

def fitsize(imgsize, max_size):


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Load pre-trained model
    model = ResidualSymmetricUNet3D(1, 3)
    model.load_state_dict(torch.load(WEIGHT))

    image = imageio.volread(DATA) #big block

    # input image is size 128**3, no overlap
    # minimun size is 16 if model is depth of 4


    list_aff = []
    for pos in glid:
        x, x_, y, y_, z, z_ = pos
        img = image[x:x_, y,y_, z,z_]

        # Predict affinity maps
        with torch.no_grad():
            input = torch.tensor(img, dtype=torch.float16).unsqueeze(0)
            affinity_map = model(input)
        
        list_aff.append(affinity_map)

    
    np.save(affinity_maps, "../pred/test.npy")