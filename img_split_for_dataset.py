from PIL import Image
import os
import numpy as np
import csv
import random

img_path = "image/train-input.tif"
label_path = "image/train-labels.tif"


def load_image_slices(file_path):
    with Image.open(file_path) as img:
        slices = []
        for i in range(img.n_frames):
            img.seek(i)
            slices.append(np.array(img))
    return np.stack(slices)

def split_into_cubes(image, cube_size=64):
    cubes = []
    z, y, x = image.shape
    for i in range(0, z, cube_size):
        for j in range(0, y, cube_size):
            for k in range(0, x, cube_size):
                cube = image[i:i+cube_size, j:j+cube_size, k:k+cube_size]
                if cube.shape == (cube_size, cube_size, cube_size):
                    cubes.append(cube)
    return cubes

def save_cubes(cubes, base_path):
    paths = []
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for idx, cube in enumerate(cubes):
        _path =  f'cube_{idx}.tiff'
        path = os.path.join(base_path, _path)
        # Save each cube slice as a separate frame in the TIFF file
        with Image.fromarray(cube[0]) as img:
            img.save(path, save_all=True, append_images=[Image.fromarray(cube[z]) for z in range(1, cube.shape[0])])
        paths.append(_path)
    return paths

def generate_csv(index, paths1, paths2, csv_path='data/fold.csv'):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'img', 'label', 'fold'])
        for idx in range(index):
            fold_number = random.randint(0, 6)
            writer.writerow([idx, paths1[idx], paths2[idx], fold_number])

# Main script
img = load_image_slices(img_path)
label = load_image_slices(label_path)

img_cube = split_into_cubes(img)
label_cube = split_into_cubes(label)

paths1 = save_cubes(img_cube, 'data/img/')
paths2 = save_cubes(label_cube, 'data/label/')

generate_csv(len(img_cube), paths1, paths2)
