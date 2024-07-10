import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dataset import ConnectomicsDataset
from model import ResidualSymmetricUNet3D
import imageio

MAX_SIZE = (128,128,128)
DATA = "image/test-input.tif"
WEIGHT  = 'residual_symmetric_unet3d.pth'
SAVE_PATH = "../pred/test.npy"

#mininum ver for x,y > MAX_SIZE
def fitsize(imgsize, max_size): # minimun size is 16 if model is depth of 4
    x, y, z = imgsize
    X, Y, Z = max_size
    nz = z // Z + 1
    sizez = (z // nz // 16) * 16 #all vertecies need to be 16N
    ny = y // Y 
    nx = x // X
    glid = []
    px, py, pz = 0, 0, 0
    for _ in range(nz):
        for _ in range(ny):
            for _ in range(nx):
                glid.append((px, px + X, py, py + Y, pz, pz + sizez))
                px += X
            py += Y
            px = 0
        pz += sizez
        py = 0
    return glid

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    # Load pre-trained model
    model = ResidualSymmetricUNet3D(1, 3)
    model.load_state_dict(torch.load(WEIGHT))

    image = imageio.volread(DATA) #big block
    imgshape = image.shape
    glid = fitsize(imgshape, MAX_SIZE)
    # input image is size 128**3, no overlap

    arr = np.zeros((3,) + imgshape, dtype=np.uint8)
    for pos in glid:
        x, x_, y, y_, z, z_ = pos
        img = image[x:x_, y,y_, z,z_]

        # Predict affinity maps
        with torch.no_grad():
            input = torch.tensor(img, dtype=torch.float16).unsqueeze(0)
            affinity_map = model(input)
        
        arr[:, x:x_, y,y_, z,z_] = affinity_map

    
    np.save("../pred/test.npy", arr)