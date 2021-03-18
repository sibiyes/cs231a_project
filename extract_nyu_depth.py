import os
import sys

import h5py
import scipy.io as sio
import skimage.io as io
from skimage import color
import numpy as np
import tensorflow_datasets as tfds

script_folder = os.path.dirname(os.path.abspath(__name__))
base_folder = os.path.dirname(script_folder)

def load_matlab_content():
    nyu_depth_file = base_folder + '/data_samples/nyu_depth_v2_labeled.mat'
    # contents = sio.loadmat(nyu_depth_file)
    # print(contents)
    
    # read mat file
    f = h5py.File(nyu_depth_file)

    # read 0-th image. original format is [3 x 640 x 480], uint8
    img = f['images'][0]

    # reshape
    img_ = np.empty([480, 640, 3])
    img_[:,:,0] = img[0,:,:].T
    img_[:,:,1] = img[1,:,:].T
    img_[:,:,2] = img[2,:,:].T

    # imshow
    img__ = img_.astype('float32')
    io.imshow(img__/255.0)
    io.show()


    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    depth = f['depths'][0]

    # reshape for imshow
    depth_ = np.empty([480, 640, 3])
    depth_[:,:,0] = depth[:,:].T
    depth_[:,:,1] = depth[:,:].T
    depth_[:,:,2] = depth[:,:].T
    
    depth1d = np.empty([480, 640])
    depth1d = depth[:,:].T
    
    print(depth1d)
    print(np.shape(depth1d))
    
    depth_rgb = color.gray2rgb(depth1d/4.0)
    
    io.imshow(depth_rgb)
    io.show()

    io.imshow(depth_/4.0)
    io.show()
    
def load_data_individual():
    data_folder = base_folder + '/data_samples/basements/basement_0001a'
    
    for f in os.listdir(data_folder):
        print(f)
        
def load_data_tf():
    ds = tfds.load('nyu_depth_v2', split='train')
    for ex in ds.take(4):
        print(ex)
        
def main():
    load_matlab_content()
    #load_data_individual()
    #load_data_tf()
    
if __name__ == '__main__':
    main()
