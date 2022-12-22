import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import io


if __name__ == '__main__':
    dtm = io.imread('G:/dem_image/dtm/001336_1560-DTM_1.tif', plugin='pil')
    ori = io.imread('G:/dem_image/ori/001336_1560-ORI_1.tif', plugin='pil')
    dtm = np.array(dtm, dtype=np.float32)
    ori = np.array(ori[:, :, 0], dtype=np.float32)
    dtm /= 255.
    ori /= 255.
    
    io.imshow(dtm)
    plt.show()
    io.imshow(ori)
    plt.show()
    print(dtm)
    print(ori)
