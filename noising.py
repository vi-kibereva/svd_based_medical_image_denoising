import numpy as np

def add_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Adds Additive White Gaussian Noise (AWGN) to the image.
    
    Sources: MRI sensors, electronic amplifiers, and power supply fluctuations.
    Cause: Thermal vibrations of electrons in sensors and voltage variations.
    Physics: The noise is independent of the image intensity (additive).
    Clinical Importance: Cleaning this noise allows for higher quality scans 
    from older or more affordable hardware, matching premium equipment performance.
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


#
# def add_poisson_noise(image: np.ndarray) -> np.ndarray:
#     """
#     Adds Poisson Noise (Shot Noise) to the image.
    
#     Sources: X-ray machines, CT (computed tomography), and PET scanners.
#     Cause: Arises due to the discrete nature of radiation and photon counting.
#     Physics: The noise is signal-dependent; rays hit the detector unevenly.
#     Clinical Importance: Efficiently cleaning this noise allows medical 
#     professionals to expose patients to lower radiation doses while 
#     maintaining diagnostic clarity.
#     """
#     vals = len(np.unique(image))
#     vals = 2 ** np.ceil(np.log2(vals))
#     noisy = np.random.poisson(image * vals) / float(vals)
#     return np.clip(noisy, 0, 255).astype(np.uint8)


# This kind of noise can't be cleaned by SVD/ But can be fixed by using median filter
# def add_salt_and_pepper(image: np.ndarray, prob: float = 0.05) -> np.ndarray:
#     """
#     Adds Salt and Pepper (Impulse) Noise to the image.
    
#     Sources: Faulty digital sensors, memory errors, or data transmission (PACS).
#     Cause: Sharp, sudden disturbances in the signal causing pixel saturation.
#     Physics: Randomly replaces original pixels with extreme values (0 or 255).
#     Clinical Importance: Removing these artifacts prevents "false positive" 
#     diagnoses, where a black/white dot might be mistaken for a micro-calcification.
#     """
#     noisy = image.copy()
#     thres = 1 - prob
#     random_matrix = np.random.random(image.shape)
#     noisy[random_matrix < prob] = 0
#     noisy[random_matrix > thres] = 255
#     return noisy




# Can be managed but needs log transformation.
# def add_speckle_noise(image: np.ndarray, sigma: float = 0.1) -> np.ndarray:
#     """
#     Adds Multiplicative Speckle Noise to the image.
    
#     Sources: Ultrasound (US) and Optical Coherence Tomography (OCT).
#     Cause: Interference of waves reflecting off small cellular structures.
#     Physics: A multiplicative noise that scales with the signal intensity.
#     Clinical Importance: Vital for automated AI diagnostic systems to clearly 
#     distinguish organ boundaries from grainy "noise textures."
#     """
#     noise = np.random.randn(*image.shape) * sigma
#     noisy = image + image * noise
#     return np.clip(noisy, 0, 255).astype(np.uint8)