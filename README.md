# SVD-Based Medical Image Denoising

This project implements a medical image denoising system based on Singular Value Decomposition (SVD) and Non-Local Block Matching. It is designed to handle various types of noise commonly found in medical imaging, such as Gaussian and Poisson noise, improving diagnostic clarity and scan quality.

## Features

- **Multi-Noise Support:** Includes modules to simulate Gaussian, Poisson, Salt & Pepper, and Speckle noise.
- **Advanced Denoising Pipeline:**
  - **Noise Estimation:** Automatically estimates noise standard deviation using Wavelet decomposition (Median Absolute Deviation of HH sub-band).
  - **Block Matching:** Parallelized grouping of similar patches across the image to leverage non-local redundancy.
  - **SVD Filtering:** Applies SVD to patch groups and uses adaptive thresholding on singular values to separate signal from noise.
  - **Weighted Reconstruction:** Reconstructs the image from patches using a 2D Gaussian mask to reduce boundary artifacts.
- **Performance Optimized:** Utilizes `joblib` for parallel processing of image blocks.
- **Benchmarking Suite:** Includes tools to measure PSNR and SSIM metrics across image datasets.

## Algorithm Overview

1.  **Noise Estimation:** Uses the Median Absolute Deviation (MAD) of the highest frequency wavelet sub-band (HH) to estimate the noise standard deviation ($\sigma$).
2.  **Block Matching:** For each block in the noisy image, it searches a local window for similar blocks based on L2 distance. These similar blocks are stacked into a matrix.
3.  **SVD Denoising:** Each group of similar blocks (matrix $M$) is decomposed using SVD ($M = U\Sigma V^T$). Singular values below an adaptive threshold (proportional to estimated $\sigma$) are zeroed out.
4.  **Aggregation:** The denoised blocks are returned to their original positions and averaged using a Gaussian weighting mask to produce the final smooth image.

## Getting Started

### Prerequisites

- Python 3.8+
- [Pip](https://pip.pypa.io/en/stable/installation/)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/svd_based_medical_image_denoising.git
    cd svd_based_medical_image_denoising
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r scripts/requirements.txt
    ```

## Usage

### Single Image Demonstration
To run the main denoising demonstration:
```bash
python scripts/main.py
```
This will load a sample image, add synthetic noise, perform denoising, and save the results in the `assets/` directory.

### Running Benchmarks
To evaluate the algorithm's performance on a dataset:
```bash
python scripts/tests.py
```
This script calculates **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) for images in `assets/test_set/` and saves the results to `denoising_benchmark_results.csv`.

### Configuration
You can adjust parameters in the scripts:
- `BLOCK_SIZE`: Size of the square patches (default: 4-6).
- `WINDOW_SIZE`: Size of the search window for block matching.
- `DISTANCE_THRESHOLD`: Sensitivity of block similarity matching.
- `SAFETY_COEFFICIENT`: Multiplier for the SVD thresholding logic.

## Project Structure

- `scripts/`:
  - `main.py`: Entry point for single image demonstration.
  - `tests.py`: Benchmark suite for performance evaluation.
  - `denoising.py`: SVD-based filtering logic and thresholding.
  - `block_matching.py`: Parallelized non-local block matching implementation.
  - `noising.py`: Noise simulation (Gaussian, Poisson, etc.).
  - `puttogether.py`: Weighted patch aggregation for image reconstruction.
- `assets/`: Test datasets and output results.

## Contributing

This project was developed as part of a Linear Algebra course.

### Project Videos:
- [Viktoria Kibyeryeva](https://youtu.be/vTENsqns6AA?si=j-Xrwz_9BxQmkH6f)
- [Yuliana Vus](https://youtu.be/KlS1YpdM-I0?si=is9gCq95d0rteSXZ)
- [Olena Tumak](https://youtu.be/Dp_kqlenaB4?si=2fvTSO-89i7fcw9f)

