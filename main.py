import numpy as np
from PIL import Image
from block_matching import block_matching
from denoising import denoising
import time

PATH_TO_FILE = 'some_lung.png'
def main():
    image = Image.open(PATH_TO_FILE)
    image_matrix = np.array(image, dtype = np.int32, order = "F") # probably need to store in column major
    now = time.time()
    res = block_matching(image_matrix, 15)
    for i in res:
        print(i)
        print()
    print (time.time() - now)
    res_clean = denoising(res)
    
if __name__ == "__main__":
    main()


