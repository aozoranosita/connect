import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from dataset import ConnectomicsDataset
from model import ResidualSymmetricUNet3D

# Load pre-trained model
model = ResidualSymmetricUNet3D(1, 3)
model.load_state_dict(torch.load('pretrained_model.pth'))

# Predict affinity maps
with torch.no_grad():
    input_image = torch.from_numpy(em_image).unsqueeze(0).unsqueeze(0).float()
    affinity_maps = model(input_image)

# Step 2: Agglomerate Affinity Maps
import waterz

# Normalize affinity maps to [0, 1] range
affinities = affinity_maps.squeeze().numpy() / 255.0

# Agglomerate using waterz
seg_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
seg = waterz.agglomerate(affinities, seg_thresholds)