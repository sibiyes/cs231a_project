import os
import sys

import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def main():
    view = 'sideline'
    yard_match_template_folder = base_folder + '/images_processed/pattern_match_template/{0}/'.format(view)
    yard_match_template_gray_folder = base_folder + '/images_processed/pattern_match_template/{0}_gray/'.format(view)
    
    
    
    for yard in os.listdir(yard_match_template_folder):
        print(yard)
        if not os.path.exists(yard_match_template_gray_folder + '/' + yard):
            os.makedirs(yard_match_template_gray_folder + '/' + yard)
            
        for img_f in os.listdir(yard_match_template_folder + '/' + yard):
            image = io.imread(yard_match_template_folder + '/' + yard + '/' + img_f)
            image_gray = rgb2gray(image)
            
            #print(image_gray)
            
            # io.imshow(image_gray)
            # io.show()
        
            io.imsave(yard_match_template_gray_folder + '/{0}/{1}'.format(yard, img_f), image_gray)
        
if __name__ == '__main__':
    main()
