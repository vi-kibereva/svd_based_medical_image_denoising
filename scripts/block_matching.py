import numpy as np
import numpy.lib.stride_tricks as npw
from typing import List
from joblib import Parallel, delayed


def block_matching(matrix: np.ndarray[tuple[int, int], np.dtype[np.int32]], 
                   threshold: int, block_size: int = 5, window_size: int = 15):
    """
    Finds and groups similar patches across the image for non-local processing.
    Args:
        matrix:       The input noisy image as a 2D numpy array
        threshold:    Maximum distance to consider a block similar
        block_size:   Size of the square blocks (p x p)
        window_size:  Size of the search window around each block
    Returns:
        A 2D object array where each element is a matrix (M_noisy) of grouped patches
    """
    blocks = npw.sliding_window_view(matrix, window_shape=(block_size, block_size))
    pad_before = (window_size - block_size) // 2
    pad_after = (window_size - block_size) - pad_before
    
    pad_width = ((pad_before, pad_after), (pad_before, pad_after))
    windows = npw.sliding_window_view(np.pad(matrix, pad_width = pad_width, mode="reflect"), window_shape=(window_size, window_size)) # Make sure "reflect" is the best mode

    result = np.empty(blocks.shape[:2], dtype=object)

    with Parallel(n_jobs=-1) as parallel:
        results = parallel(
            delayed(process_column)(j , blocks, windows, threshold, block_size)
            for j in range(blocks.shape[1])
    )

    for j, column in enumerate(results):
        for i, block in enumerate(column):
            result[i, j] = block
    return result

def process_column(column_idx: int, blocks, windows, threshold, block_size):
    """
    Processes a single column of blocks for similarity matching in parallel.
    Args:
        column_idx:  Index of the column to process
        blocks:      Sliding window view of all blocks
        windows:     Sliding window view of all search windows
        threshold:   Similarity threshold
        block_size:  Size of the blocks
    Returns:
        A list of matched block matrices for the column
    """
    column_results = []
    for i in range(blocks.shape[0]):
        column_results.append(match_block(blocks[i, column_idx], windows[i, column_idx], threshold, block_size))
    return column_results


def match_block(target: np.ndarray, window: np.ndarray, threshold: int, block_size: int = 5) -> np.ndarray:
    """
    Finds blocks within a search window that are similar to the target block.
    Args:
        target:      The reference block to match
        window:      The local search window
        threshold:   Similarity threshold (squared L2 distance)
        block_size:  Size of the blocks
    Returns:
        M_noisy: A matrix where each column is a flattened similar block
    """
    candidates = npw.sliding_window_view(window, window_shape=(block_size, block_size))
    
    diff = candidates - target
    dist = np.mean(np.square(diff), axis=(-2, -1))
    
    mask = dist < threshold ** 2
    matched_blocks = candidates[mask]
    
    target_col = target.reshape(-1, 1)
    
    if matched_blocks.shape[0] == 0:
        return target_col
        
    matches_cols = matched_blocks.reshape(matched_blocks.shape[0], -1).T
    
    M_noisy = np.hstack((target_col, matches_cols))
    
    return M_noisy
