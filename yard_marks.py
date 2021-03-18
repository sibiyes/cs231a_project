import os
import sys

import numpy as np

from skimage.filters.rank import gradient
from skimage.color import rgb2gray, rgb2hsv
from skimage.morphology import disk, square
from skimage.morphology import binary_dilation, binary_erosion, binary_closing
from skimage.feature import canny
from skimage.morphology import binary_closing, binary_opening

from scipy.ndimage.morphology import binary_fill_holes

from skimage.feature import blob_dog, blob_log, blob_doh

import matplotlib.pyplot as plt
from skimage import io
from matplotlib import cm

from scipy.ndimage.morphology import binary_fill_holes

from utils import load_sideline_mask

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)




def yard_mark_detection(image, img_f, view):
    
    image_gray = rgb2gray(image)
    # io.imshow(image_gray)
    # io.show()
    
    if (view == 'sideline'):
        sideline_mask = load_sideline_mask(img_f)
        sideline_mask = sideline_mask.astype(int)
    else:
        sideline_mask = np.ones_like(image_gray)
    
    print('sideline mask')
    print(sideline_mask)
    
    grad = gradient(image_gray, disk(2), mask = sideline_mask)
    grad = grad/255
    
    print(np.min(grad), np.max(grad), np.mean(grad))
    
    
    grad_filter_high = np.zeros_like(image_gray)
    grad_filter_high[grad > 0.5] = 1
    
    grad_filter_mid = np.zeros_like(image_gray)
    grad_filter_mid[np.logical_and(grad > 0.2, grad < 0.5)] = 1
    
    # io.imshow(grad_filter_high)
    # io.show()
    
    
    grad_high_erode = binary_erosion(grad_filter_high, disk(1))
    grad_high_dilate  = binary_dilation(grad_high_erode, disk(5))
    #grad_high_dilate = iterative_dilate(grad_filter_high, 1)
    
    # io.imshow(grad_high_dilate)
    # io.show()
    
    mask2 = np.logical_and(sideline_mask, np.logical_not(grad_high_dilate)).astype(int)
    
    # io.imshow(mask2)
    # io.show()
    
    ### step 2 with updated mask
    
    grad = gradient(image_gray, disk(2), mask = mask2)
    grad = grad/255
    
    print(np.min(grad), np.max(grad), np.mean(grad))
    
    # io.imshow(grad)
    # io.show()

    #return

    grad_filter_high = np.zeros_like(image_gray)
    grad_filter_high[grad > 0.5] = 1
    
    grad_filter_mid = np.zeros_like(image_gray)
    grad_filter_mid[np.logical_and(grad > 0.2, grad < 0.5)] = 1
    
    # io.imshow(grad_filter_mid)
    # io.show()
    
    ### edges and closing
    
    edges = canny(grad_filter_mid, 1.0)
    #edges = canny(image_hsv[:, :, 1])
    
    # io.imshow(edges)
    # io.show()
    
    edges_closed = binary_closing(edges)
    # io.imshow(edges_closed)
    # io.show()
    
    edges_filled = binary_fill_holes(edges_closed)
    edges_filled = binary_opening(edges_filled)
    # io.imshow(edges_filled)
    # io.show()
    
    
    #####################################
    
    blob_type = 'dog'
    
    if (blob_type == 'log'):
        blobs = blob_log(edges_filled, max_sigma=100, num_sigma=10, threshold=.1)
    if (blob_type == 'dog'):
        blobs = blob_dog(edges_filled, max_sigma=100, threshold=.1)
    if (blob_type == 'doh'):
        blobs = blob_doh(edges_filled, max_sigma=100, threshold=.01)
    
    print('blobs')
    print(blobs)
    print(np.shape(blobs))
    
    img_h, img_w = np.shape(image_gray)
    
    print(min(blobs[:, 2]), max(blobs[:, 2]), np.mean(blobs[:, 2]))
    
    
    blobs = blobs[blobs[:, 2] > 20]
    blobs = blobs[np.logical_or((blobs[:, 0] > 0.7*img_h), (blobs[:, 0] < 0.3*img_h))]
    
    
    blob_centres = np.zeros_like(grad_filter_high)
    print(np.shape(blob_centres))
    for xs, ys, r in blobs:
        blob_centres[int(xs), int(ys)] = 1
        
    blob_centres = blob_centres.astype(float)
    # io.imshow(blob_centres)
    # io.show()
    
    #return
    
    # blob_centre_output_folder = base_folder + '/images_processed/blob_centre/{0}/{1}'.format(view, blob_type)
    # if not os.path.exists(blob_centre_output_folder):
    #     os.makedirs(blob_centre_output_folder)
        
    #io.imsave(blob_centre_output_folder + '/' + img_f, blob_centres)
    
    ### plot blobs
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    ax = axes.ravel()
    
    ax[0].imshow(image)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].imshow(image)
    origin = np.array((0, image.shape[1]))
    
    img_subs = []
    for xs, ys, r in blobs:
        
        #ax[1].plot(xs, ys, 'b')
        
        
        c = plt.Circle((ys, xs), r, color='red', linewidth=2, fill=False)
        ax[1].add_patch(c)
        
        w_sub = 100
        xs_sub = [max(0, int(xs-w_sub)), min(img_h, int(xs+w_sub))]
        ys_sub = [max(0, int(ys-w_sub)), min(img_w, int(ys+w_sub))]
        print('xy', xs, ys)
        print('xs sub', xs_sub)
        print('ys sub', ys_sub)
        
        img_sub = image[xs_sub[0]:xs_sub[1], ys_sub[0]:ys_sub[1], :]
        print('img_sub', np.shape(img_sub))
        img_subs.append(img_sub)
        
        
    ax[1].set_xlim(origin)
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    #ax[1].set_title(title)
    
    plt.tight_layout()
    
    plt.show()
    
    for img_sub in img_subs:
        print(np.shape(img_sub))
        io.imshow(img_sub)
        io.show()
    
    # blob_output_folder = base_folder + '/images_processed/blobs/{0}/{1}'.format(view, blob_type)
    # if not os.path.exists(blob_output_folder):
    #     os.makedirs(blob_output_folder)
    # 
    # #plt.gcf().set_size_inches(20, 15)
    # plt.savefig(blob_output_folder + '/' + img_f)
