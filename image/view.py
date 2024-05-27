from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_3d_tiff(file_path):
    with Image.open(file_path) as img:
        slices = []
        for i in range(img.n_frames):
            img.seek(i)
            slices.append(np.array(img))
    return np.stack(slices)

def compute_statistics(image):
    min_val = np.min(image)
    max_val = np.max(image)
    histogram, bin_edges = np.histogram(image, bins=256, range=(min_val, max_val))
    return min_val, max_val, histogram, bin_edges

def plot_histogram(histogram, bin_edges):
    plt.figure()
    plt.title("Histogram")
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.stem(bin_edges[:-1], histogram, linefmt='-', markerfmt=' ', basefmt=' ')
    plt.show()

if __name__ == '__main__':
    file_path = 'train-labels.tif'
    image = load_3d_tiff(file_path)
    
    min_val, max_val, histogram, bin_edges = compute_statistics(image)
    
    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")
    
    #plot_histogram(histogram, bin_edges)
