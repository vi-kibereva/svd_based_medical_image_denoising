import numpy as np
from typing import List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


def denoising(res:  np.ndarray[tuple[int, int], np.dtype[np.object_]], noise_tresh: float = 850.0): #treshhold probably should be counted energybased
    res_clean = np.empty_like(res)
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            M = res[i, j]
            U, Sigma, Vt = np.linalg.svd(M, full_matrices=False) # це якщо модна з бібліотеки взяти 
            Sigma[Sigma < noise_tresh] = 0
            # Sigma[3:] = 0
            # res_clean[i, j] = (U * Sigma) @ Vt
            res_clean[i, j] = (U @ np.diag(Sigma)) @ Vt

    return res_clean

# def denoising(res: np.ndarray, energy_ratio: float = 0.7): 
#     res_clean = np.empty_like(res)
    
#     for i in range(res.shape[0]):
#         for j in range(res.shape[1]):
#             M = res[i, j]
#             U, Sigma, Vt = np.linalg.svd(M, full_matrices=False)
            
#             total_energy = np.sum(Sigma ** 2)
            
#             cumulative_energy = np.cumsum(Sigma ** 2)
            
#             mask = cumulative_energy <= (total_energy * energy_ratio)
            
#             mask[0] = True 
            
#             Sigma[~mask] = 0
            
#             res_clean[i, j] = (U @ np.diag(Sigma)) @ Vt

#     return res_clean

# def denoising(res:  np.ndarray[tuple[int, int], np.dtype[np.object_]], noise_tresh: float = 50.0): #treshhold probably should be counted energybased
#     original_shape = res.shape
#     flat_res = [res[i, j] for i in range(res.shape[0]) for j in range(res.shape[1])]
#     denoised_flat = Parallel(n_jobs=-1)(
#         delayed(_process_single_matrix)(M, noise_tresh, svd) 
#         for M in flat_res
#     )
#     res_clean = np.array(denoised_flat, dtype=object).reshape(original_shape)
#     return res_clean

# def _process_single_matrix(M, noise_tresh, svd_func):
#     U, Sigma, Vt = svd_func(M)
#     Sigma[Sigma < noise_tresh] = 0
#     return (U * Sigma) @ Vt

# #Using np.linalg.eig for eigen decomposition if it is allowed
# # def svd(M):
# #     A = M.T @ M
# #     w, V = np.linalg.eig(A)
# #     idx = np.argsort(w)[::-1]
# #     w = w[idx]
# #     V = V[:, idx]

# #     Sigma = np.sqrt(np.maximum(w.real, 0))
    
# #     safe_sigma = np.where(Sigma == 0, 1e-10, Sigma)
# #     U = (M @ V) / safe_sigma
# #     return U, Sigma, V.T

# def svd(M: np.dtype[np.object_]):
#     A = M.T @ M

#     vals, V = eigen_decomposition(A)
#     Sigma = np.sqrt(vals)

#     safe_sigma = np.where(Sigma == 0, 1e-10, Sigma)
#     U = (M @ V) / safe_sigma
#     return U, Sigma, V.T

# def eigen_decomposition(A, iterations = 60):
#     n = A.shape[0]
#     V = np.eye(n)
#     A_k = A.copy().astype(np.float64)

#     for _ in range(iterations):
#         Q, R = qr_decomposition(A_k)
#         A_k = R @ Q
#         V = V @ Q

#     eigenvalues = np.diag(A_k)

#     idx = eigenvalues.argsort()[::-1]
#     return eigenvalues[idx], V[:, idx]

# def qr_decomposition(A):
#     n, m = A.shape
#     Q = np.zeros((n, m))
#     R = np.zeros((m, m))

#     for j in range(m):
#         v = A[:, j].copy()
#         for i in range(j):
#             R[i, j] = Q[:, i] @ v
#             v = v - R[i, j] * Q[:, i]

#         R[j, j] = np.linalg.norm(v)
#         Q[:, j] = v / (R[j, j] + 1e-10)

#     return Q, R