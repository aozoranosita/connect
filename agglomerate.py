import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

def agglomerate(affinity_map, threshold=0.5, min_size=100):
    # Convert affinity map to distance map for watershed
    distance = np.linalg.norm(affinity_map, axis=0)
    
    # Apply threshold to get initial binary mask
    mask = distance > threshold
    
    # Remove small objects
    mask = remove_small_objects(mask, min_size=min_size)
    
    # Compute distance transform
    distance_transform = ndi.distance_transform_edt(mask)
    
    # Find local maxima
    local_maxi = peak_local_max(distance_transform, indices=False, labels=mask)
    
    # Generate markers for watershed
    markers, _ = ndi.label(local_maxi)
    
    # Perform watershed segmentation
    labels = watershed(-distance_transform, markers, mask=mask)
    
    return labels

# Example usage
affinity_map = affinity_map.detach().cpu().numpy()
segmentation = agglomerate(affinity_map[0])
