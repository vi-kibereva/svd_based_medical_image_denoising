import numpy as np

def aggregate_patches(res_clean, image_shape, patch_size=5):
    """
    Reconstructs the full image from denoised patch groups.

    Args:
        denoised_groups: list of M_clean matrices, one per reference patch.
                         Each M_clean has shape (patch_size^2, num_similar_patches)
        patch_coords:    list of lists of (row, col) tuples — for each group,
                         the top-left coordinates of every patch in that group
        image_shape:     (H, W) of the original image
        patch_size:      p (patches are p x p)

    Returns:
        denoised image as a 2D numpy array
    """
    H, W = image_shape
    image_canvas = np.zeros((H, W), dtype=np.float64)
    weight_canvas = np.zeros((H, W), dtype=np.float64)

    for i in range(res_clean.shape[0]):
        for j in range(res_clean.shape[1]):
            M_clean = res_clean[i, j]  # shape: (patch_size^2, num_similar_patches)

            if i + patch_size > H or j + patch_size > W:
                continue

            # average all similar denoised patches, reshape to 2D
            patch_2d = M_clean.mean(axis=1).reshape(patch_size, patch_size)  # default C-order, no order argument needed

            image_canvas[i:i+patch_size, j:j+patch_size] += patch_2d
            weight_canvas[i:i+patch_size, j:j+patch_size] += 1.0

    weight_canvas[weight_canvas == 0] = 1.0
    denoised_image = image_canvas / weight_canvas
    return denoised_image
# Maybe add gaussian mask, it says it is better        

