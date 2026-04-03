import numpy as np
from PIL import Image
from block_matching import block_matching
from denoising import denoising
from puttogether import aggregate_patches
from noising import add_gaussian_noise
import time
#import pywt
#import cv2
import matplotlib.pyplot as plt


PATH_TO_FILE = 'Some_lung.png'
BLOCK_SIZE = 8
WINDOW_SIZE = 31
SVD_THRESHOLD = 0.99
DISTANCE_THRESHOLD = 10


def estimate_noise_sigma(image_matrix):
    coeffs = pywt.dwt2(image_matrix, 'db2')
    LL, (LH, HL, HH) = coeffs
    
    mad = np.median(np.abs(HH - np.median(HH)))
    
    return mad / 0.6745

PATH_TO_FILE = 'COVID-1017.png'
def main():
    image = Image.open(PATH_TO_FILE)
    image_matrix = np.array(image, dtype = np.int32, order = "F") # probably need to store in column major
    # image_matrix = cv2.imread(PATH_TO_FILE, cv2.IMREAD_GRAYSCALE)


    # #take small part
    # h, w = image_matrix.shape
    # image_matrix = image_matrix[h-150:h-50, w-150:w-50].copy()
    # print(f"Working on fragment size: {image_matrix.shape}")
    # Image.fromarray(image_matrix).save("test_fragment.png")
    # #take small part end 

    # image_matrix_noisy = add_poisson_noise(image_matrix)
    image_matrix_noisy = add_gaussian_noise(image_matrix, 10)

    Image.fromarray(image_matrix_noisy).save("noisy_test_fragment.png")

    estimated_sigma = estimate_noise_sigma(image_matrix_noisy)
    print(f"Estimated sigma: {estimated_sigma:.2f}")
    
    print("Starting Block Matching...")

    now = time.time()

    res = block_matching(image_matrix_noisy, threshold = DISTANCE_THRESHOLD, block_size=BLOCK_SIZE, window_size=WINDOW_SIZE)

    print(f"Block Matching took: {time.time() - now:.2f}s")

    print("Starting SVD Denoising...")
    svd_start = time.time()


    res_clean = denoising(res, SVD_THRESHOLD)

    print(f"SVD Denoising took: {time.time() - svd_start:.2f}s")

    clean_image = aggregate_patches(res_clean, image_matrix.shape, BLOCK_SIZE)

    clean_image = np.clip(clean_image, 0, 255).astype(np.uint8)
    Image.fromarray(clean_image).save("denoised_result.png")
    print("Saved to denoised_result.png")

    print(f"Total time: {time.time() - now:.2f}s")

if __name__ == "__main__":
    main()


