import numpy as np
from joblib import Parallel, delayed

def denoising(res:  np.ndarray[tuple[int, int], np.dtype[np.object_]], sigma,  safety_coefficient): #treshhold probably should be counted energybased
    """
    Applies SVD-based denoising to each group of similar patches.
    Args:
        res:                2D object array of noisy patch matrices
        sigma:              Estimated noise standard deviation
        safety_coefficient: Multiplier for the thresholding logic
    Returns:
        res_clean: 2D object array of denoised patch matrices
    """
    res_clean = np.empty_like(res)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            M = res[i, j]
            U, Sigma, Vt = np.linalg.svd(M, full_matrices=False)

            N_patches = M.shape[0]
            Patch_size_squared = M.shape[1]
            
            thresh = sigma * np.sqrt(max(N_patches, Patch_size_squared))
            
            thresh *= safety_coefficient

            safe_indices = 1
            
            for idx in range(safe_indices, len(Sigma)):
                if Sigma[idx] < thresh:
                    Sigma[idx] = 0

            res_clean[i, j] = (U @ np.diag(Sigma)) @ Vt

    return res_clean

def custom_denoising(res:  np.ndarray[tuple[int, int], np.dtype[np.object_]], noise_tresh: float = 50.0): #treshhold probably should be counted energybased
    original_shape = res.shape
    flat_res = [res[i, j] for i in range(res.shape[0]) for j in range(res.shape[1])]
    denoised_flat = Parallel(n_jobs=-1)(
        delayed(_process_single_matrix)(M, noise_tresh, svd)
        for M in flat_res
    )
    res_clean = np.empty(original_shape, dtype=object)
    for i, patch in enumerate(denoised_flat):
        res_clean.flat[i] = patch.real
    return res_clean

def _process_single_matrix(M, noise_tresh, svd_func):
    U, Sigma, Vt = svd_func(M)
    Sigma[Sigma < noise_tresh] = 0
    return (U * Sigma) @ Vt

# #Using np.linalg.eig for eigen decomposition if it is allowed
def svd(M):
    A = M.T @ M
    w, V = np.linalg.eig(A)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    Sigma = np.sqrt(np.maximum(w.real, 0))
    
    safe_sigma = np.where(Sigma == 0, 1e-10, Sigma)
    U = (M @ V) / safe_sigma
    return U, Sigma, V.T

def custom_svd(M: np.dtype[np.object_]):
    A = M.T @ M

    vals, V = eigen_decomposition(A)
    Sigma = np.sqrt(np.maximum(vals, 0))

    safe_sigma = np.where(Sigma == 0, 1e-10, Sigma)
    U = (M @ V) / safe_sigma
    return U, Sigma, V.T

def eigen_decomposition(A, iterations = 60):
    n = A.shape[0]
    V = np.eye(n)
    A_k = A.copy().astype(np.float64)

    for _ in range(iterations):
        Q, R = qr_decomposition(A_k)
        A_k = R @ Q
        V = V @ Q

    eigenvalues = np.diag(A_k)

    idx = eigenvalues.argsort()[::-1]
    return eigenvalues[idx], V[:, idx]

def qr_decomposition(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / (R[j, j] + 1e-10)

    return Q, R
