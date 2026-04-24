import numpy as np
from scipy.signal.windows import gaussian

def aggregate_patches(res_clean, image_shape, patch_size=5):
    """
    Reconstructs the full image from denoised patch groups.
    Args:
        res_clean:    2D object array of denoised patch matrices,
                      each of shape (patch_size^2, num_similar_patches)
        image_shape:  (H, W) of the original image
        patch_size:   p (patches are p x p)
    Returns:
        denoised image as a 2D numpy array
    """
    H, W = image_shape
    image_canvas = np.zeros((H, W), dtype=np.float64)
    weight_canvas = np.zeros((H, W), dtype=np.float64)

    # build 2D Gaussian mask once, reuse for every patch
    g = gaussian(patch_size, std=patch_size / 4)
    gaussian_mask = np.outer(g, g)  # shape (patch_size, patch_size)

    for i in range(res_clean.shape[0]):
        for j in range(res_clean.shape[1]):
            M_clean = res_clean[i, j]  # shape: (patch_size^2, num_similar_patches)

            patch_2d = M_clean.mean(axis=1).reshape(patch_size, patch_size)

            image_canvas[i:i+patch_size, j:j+patch_size] += patch_2d * gaussian_mask
            weight_canvas[i:i+patch_size, j:j+patch_size] += gaussian_mask

    weight_canvas[weight_canvas == 0] = 1.0
    denoised_image = image_canvas / weight_canvas
    return denoised_image