import numpy as np

def compute_affinity_maps_3d(segmentation_labels):
    """
    Compute affinity maps from 3D segmentation labels.
    
    Parameters:
    segmentation_labels (numpy.ndarray): 3D array of segmentation labels.
    
    Returns:
    numpy.ndarray: 4D array of affinity maps with shape (3, depth, height, width).
                   The first channel corresponds to affinities in the x direction,
                   the second channel corresponds to affinities in the y direction,
                   and the third channel corresponds to affinities in the z direction.
    """
    # Initialize the affinity maps
    depth, height, width = segmentation_labels.shape
    affinity_maps = np.zeros((3, depth, height, width), dtype=np.float32)
    
    # Compute affinities in the x direction
    affinity_maps[0, :, :, :-1] = (segmentation_labels[:, :, :-1] == segmentation_labels[:, :, 1:]).astype(np.float32)
    
    # Compute affinities in the y direction
    affinity_maps[1, :, :-1, :] = (segmentation_labels[:, :-1, :] == segmentation_labels[:, 1:, :]).astype(np.float32)
    
    # Compute affinities in the z direction
    affinity_maps[2, :-1, :, :] = (segmentation_labels[:-1, :, :] == segmentation_labels[1:, :, :]).astype(np.float32)
    
    return affinity_maps

if __name__ == "__main__":
    # Example usage
    segmentation_labels = np.array([
        [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2]
        ],
        [
            [1, 1, 0, 0],
            [1, 1, 0, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2]
        ]
    ])

    affinity_maps = compute_affinity_maps_3d(segmentation_labels)
    print("Affinity map in x direction:\n", affinity_maps[0])
    print("Affinity map in y direction:\n", affinity_maps[1])
    print("Affinity map in z direction:\n", affinity_maps[2])
