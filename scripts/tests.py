import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from block_matching import block_matching
from denoising import denoising
from puttogether import aggregate_patches
from noising import add_gaussian_noise

IMAGE_DIR = 'assets/test_set/'
BLOCK_SIZE = 4
WINDOW_SIZE = 21
DISTANCE_THRESHOLD = 9
SAFETY_COEFFICIENT = 1.5
NOISE_SIGMA = 25

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


def process_single_image(file_path):
    image = Image.open(file_path).convert('L')
    gt = np.array(image, dtype=np.float64)
    noisy = np.clip(add_gaussian_noise(gt.copy(), NOISE_SIGMA), 0, 255)
    estimated_sigma = estimate_noise_sigma(noisy)

    start = time.time()
    res = block_matching(noisy, threshold=DISTANCE_THRESHOLD, block_size=BLOCK_SIZE, window_size=WINDOW_SIZE)
    res_clean = denoising(res, estimated_sigma, SAFETY_COEFFICIENT)
    clean = aggregate_patches(res_clean, gt.shape, BLOCK_SIZE)
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    elapsed = time.time() - start

    psnr_val = psnr(gt.astype(np.uint8), clean, data_range=255)
    ssim_val = ssim(gt.astype(np.uint8), clean, data_range=255)
    
    return psnr_val, ssim_val, elapsed

def main():
    results = []
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Starting benchmark on {len(image_files)} images...")

    for filename in image_files[:2]:
        path = os.path.join(IMAGE_DIR, filename)
        try:
            p, s, t = process_single_image(path)
            results.append({"File": filename, "PSNR": p, "SSIM": s, "Time": t})
            print(f"Finished {filename}: PSNR={p:.2f}, SSIM={s:.4f}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(results)
    print("\n--- Final Benchmark Summary ---")
    print(df.describe().loc[['mean', 'std']])

    df.to_csv("denoising_benchmark_results.csv", index=False)
    print("\nResults saved to denoising_benchmark_results.csv")

if __name__ == "__main__":
    main()
