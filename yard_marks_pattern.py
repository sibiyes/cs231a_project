import os
import sys

import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import match_template
from skimage.transform import rotate

import matplotlib.pyplot as plt

from analysis import get_img_files

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def yard_match(image, yard, view, img_f):
    yard_match_template_folder = base_folder + '/images_processed/pattern_match_template/{0}2/{1}'.format(view, yard)

    template_files = os.listdir(yard_match_template_folder)
    print(template_files)
    
    for yard_f in template_files:
        image_yard = io.imread(yard_match_template_folder + '/' + yard_f)
        image_yard = rgb2gray(image_yard)
        
        io.imshow(image_yard)
        io.show()
        
        
        ############################
        
        #print('shape', np.shape(image))
        result = match_template(image, image_yard)
        #print(result)
        ij = np.unravel_index(np.argmax(result), result.shape)
        x, y = ij[::-1]
        #print('xy', x, y)

        fig = plt.figure(figsize=(12, 8))
        #fig = plt.figure()
        ax1 = plt.subplot(1, 3, 1)
        ax2 = plt.subplot(1, 3, 2)
        #ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
        ax3 = plt.subplot(1, 3, 3)

        ax1.imshow(image_yard, cmap=plt.cm.gray)
        ax1.set_axis_off()
        ax1.set_title('template')

        ax2.imshow(image, cmap=plt.cm.gray)
        ax2.set_axis_off()
        ax2.set_title('image')
        # highlight matched region
        h_sub, w_sub = image_yard.shape
        #print('hw', h_sub, w_sub)
        rect = plt.Rectangle((x, y), w_sub, h_sub, edgecolor='r', facecolor='none')
        ax2.add_patch(rect)
        ax2.plot(x, y, marker='v', color="green")

        ax3.imshow(result)
        ax3.set_axis_off()
        ax3.set_title('`match_template`\nresult')
        # highlight matched region
        ax3.autoscale(False)
        ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

        plt.show()


def main():
    images_folder, image_files = get_img_files()
    
    cnt = 0
    
    for img_f in image_files:
        print(img_f)
        view = None
        if ('endzone' in img_f):
            view = 'endzone'
        else:
            view = 'sideline'
            
        if (view == 'endzone'):
            continue
            
            
        image = io.imread(images_folder + '/' + img_f)
        image_gray = rgb2gray(image)
        
        io.imshow(image_gray)
        io.show()
        
        yard = 40
        
        yard_match(image_gray, yard, view, img_f)
        
        cnt += 1
        if (cnt >= 10):
            break
        
    sys.exit(0)
        

if __name__ == '__main__':
    main()
        
        
