import os
import sys
import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.transform import rotate
from skimage.feature import canny
from skimage.filters import sobel, roberts
from skimage.draw import line
from skimage.morphology import binary_closing, binary_opening

from scipy.ndimage.morphology import binary_fill_holes

import matplotlib.pyplot as plt
from matplotlib import cm


from line_detection import run_hough, detect_yard_lines, detect_boundary
from hog_analysis import run_hog_features
from gradient_analysis import gradient_analysis
from yard_marks import yard_mark_detection


script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

    
def color_filter(image):
    image_hsv = rgb2hsv(image)
    print(image_hsv)
    print(np.shape(image_hsv))
    print(np.shape(image_hsv[:, :, 0]))
    print('H', np.min(image_hsv[:, :, 0]), np.max(image_hsv[:, :, 0]))
    print('S', np.min(image_hsv[:, :, 1]), np.max(image_hsv[:, :, 1]))
    print('V', np.min(image_hsv[:, :, 2]), np.max(image_hsv[:, :, 2]))
    
    ### Green [60 - 180]
    mask = np.zeros_like(image_hsv[:, :, 0])
    mask[ np.logical_and( 
            (image_hsv[:, :, 0] > 60/360), 
            (image_hsv[:, :, 0] < 160/360),
            (image_hsv[:, :, 1] > 0.1)
        ) ] = 1.0
    
    ### White
    # mask = np.zeros_like(image_hsv[:, :, 1])
    # mask[ (image_hsv[:, :, 1] < 0.1) ] = 1.0
    
    print( np.logical_and( (image_hsv[:, :, 0] > 60/360), (image_hsv[:, :, 0]  < 180/360) ) )
    print( np.sum(np.logical_and( (image_hsv[:, :, 0] > 60/360), (image_hsv[:, :, 0]  < 180/360) )) )
    
    # io.imshow(mask)
    # io.show()
    # 
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    # ax = axes.ravel()
    
    
    # ax[0].imshow(image)
    # ax[0].set_title('Input image')
    # ax[0].set_axis_off()
    # 
    # ax[1].imshow(mask, cmap=cm.gray)
    # ax[1].set_axis_off()
    # 
    # plt.tight_layout()
    # plt.show()
    
    return mask
    
def entity_detection(image):
    image_hsv = rgb2hsv(image)
    image = rgb2gray(image)
    
    # for i in range(3):
    #     io.imshow(image_hsv[:, :, i])
    #     io.show()
    #     
    #     edges = canny(image_hsv[:, :, i], 1.0)
    #     #edges = roberts(image)
    #     io.imshow(edges)
    #     io.show()
        
    
    edges = canny(image, 1.0)
    #edges = canny(image_hsv[:, :, 1])
    
    io.imshow(edges)
    io.show()
    
    edges_closed = binary_closing(edges)
    io.imshow(edges_closed)
    io.show()
    
    edges_filled = binary_fill_holes(edges_closed)
    edges_filled = binary_opening(edges_filled)
    io.imshow(edges_filled)
    io.show()
    
def get_img_files():
    images_folder = base_folder + '/images_extract'
    
    ### sample from all images
    image_files = os.listdir(images_folder)
    #image_files = image_files[:40]
    
    ### images with yard lines in sideview
    # sideview_img_folder = base_folder + '/images_processed/yardlines/sideline'
    # image_files = os.listdir(sideview_img_folder)
    # image_files = [f.replace('Sideline', 'Endzone') for f in image_files]
    
    
    return images_folder, image_files
    
def file_check(img_f, file_type, view = None):
    if (file_type == 'yardlines'):
        if (view is None):
            raise Exception('view argument can not be None')
        folder = base_folder + '/images_processed/yardlines/{0}'.format(view)
        if (os.path.exists(folder + '/' + img_f)):
            return True
            
    if (file_type == 'boundary'):
        folder = base_folder + '/images_processed/sideview_mask'
        if (os.path.exists(folder + '/' + img_f)):
            return True
            
    if (file_type == 'gradient'):
        folder = base_folder + '/images_processed/blob_centre/{0}'.format(view)
        if (os.path.exists(folder + '/' + img_f)):
            return True
            
    if (file_type == 'hash_marks'):
        folder = base_folder + '/images_processed/hash_marks/{0}/dog'.format(view)
        if (os.path.exists(folder + '/' + img_f)):
            return True
            
    if (file_type == 'lines_all'):
        folder = base_folder + '/images_processed/lines_all'
        if (os.path.exists(folder + '/' + img_f)):
            return True
    
        

def main():
    # images_folder = base_folder + '/nfl-impact-detection/images'
    # 
    # image_files = os.listdir(images_folder)
    # image_files = image_files[:400]
    
    images_folder, image_files = get_img_files()
    #image_files = ['57781_000252_sideline_150.jpg']
    #image_files = ['58094_000423_endzone_70.jpg']
    
    print(image_files)
    
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
            
        # if (view == 'sideline'):
        #     continue
        
        # f_check = file_check(img_f, 'yardlines', view)
        # f_check = file_check(img_f, 'boundary', view)
        # 
        # if (f_check):
        #     continue
        
            
        image = io.imread(images_folder + '/' + img_f)
        image_gray = rgb2gray(image)
        
        
        # if (view == 'sideline'):
        #     image_gray = rotate(image_gray, -90, resize = True)
        
        # io.imshow(image)
        # io.show()
        
        # io.imshow(image_gray)
        # io.show()
        
        #continue
        
        #run_hough(image, view)
        detect_boundary(image, view, img_f)
        #detect_yard_lines(image, view, img_f)
        #color_filter(image)
        #entity_detection(image)
        #run_hog_features(image)
        #gradient_analysis(image, img_f, view)
        #yard_mark_detection(image, img_f, view)
        
        #sys.exit(0)
        
        # cnt += 1
        # if (cnt >= 10):
        #     break
            
        print('-------------------------')
    
if __name__ == '__main__':
    main()
