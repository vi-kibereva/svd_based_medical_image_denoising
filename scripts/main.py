import numpy as np
from PIL import Image
from block_matching import block_matching
from denoising import denoising, custom_denoising
from puttogether import aggregate_patches
from noising import add_gaussian_noise, add_poisson_noise
import time
import pywt
import matplotlib.pyplot as plt


PATH_TO_FILE = '121.jpg'
BLOCK_SIZE = 4
WINDOW_SIZE = 15
DISTANCE_THRESHOLD = 9
SAFETY_COEFFICIENT = 1


def estimate_noise_sigma(image_matrix):
    """
    Estimates the standard deviation of Gaussian noise in an image using Wavelet decomposition.
    Args:
        image_matrix: The input image as a 2D numpy array
    Returns:
        estimated_sigma: The estimated noise standard deviation (float)
    """
    coeffs = pywt.dwt2(image_matrix, 'db2')
    LL, (LH, HL, HH) = coeffs
    
    mad = np.median(np.abs(HH - np.median(HH)))
    
    return mad / 0.6745

def main():
    image = Image.open("assets/test_set/" + PATH_TO_FILE)
    image_matrix = np.array(image, dtype = np.int32, order = "F")
    image_matrix_noisy = add_gaussian_noise(image_matrix, 10)


    Image.fromarray(image_matrix_noisy).save("assets/noisy/" + PATH_TO_FILE + ".png")

    estimated_sigma = estimate_noise_sigma(image_matrix_noisy)
    print(f"Estimated sigma: {estimated_sigma:.2f}")
    
    print("Starting Block Matching...")

    now = time.time()

    res = block_matching(image_matrix_noisy, threshold = DISTANCE_THRESHOLD, block_size=BLOCK_SIZE, window_size=WINDOW_SIZE)

    print(f"Block Matching took: {time.time() - now:.2f}s")

    print("Starting SVD Denoising...")
    svd_start = time.time()


    res_clean = custom_denoising(res, estimated_sigma * SAFETY_COEFFICIENT)

    print(f"SVD Denoising took: {time.time() - svd_start:.2f}s")

    clean_image = aggregate_patches(res_clean, image_matrix.shape, BLOCK_SIZE)

    clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
    Image.fromarray(clean_image).save("assets/results/denoised" + PATH_TO_FILE + ".png")
    print("Saved to results dir")

    print(f"Total time: {time.time() - now:.2f}s")

if __name__ == "__main__":
    main()


