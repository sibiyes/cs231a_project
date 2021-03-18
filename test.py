import os
import sys

import numpy as np

from skimage import io
from skimage.util import img_as_ubyte, img_as_uint

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def main():
    print(5)
    
    img0 = np.zeros((400, 400)).astype(int).astype(float)
    #img0 = img_as_uint(img0)
    img1 = np.ones((400, 400)).astype(int).astype(float)
    #img1 = img_as_uint(img1)
    
    io.imshow(img0)
    io.show()
    
    io.imshow(img1)
    io.show()
    
if __name__ == '__main__':
    main()
