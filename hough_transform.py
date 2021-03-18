### https://alyssaq.github.io/2014/understanding-hough-transform/

import os
import sys

import numpy as np
#import imageio
import math

import matplotlib.pyplot as plt
from matplotlib import cm

from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import rotate

from line_detection import run_hough, plot_lines
from analysis import file_check

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

# def rgb2gray(rgb):
#     return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


def hough_line(img, angle_step=1, lines_are_white=True, value_threshold=5):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    lines_are_white - boolean indicating whether lines to be detected are white
    value_threshold - Pixel values above or below the value_threshold are edges
    Returns:
    accumulator - 2D array of the hough transform accumulator
    theta - array of angles used in computation, in radians.
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image.
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    
    # thetas = np.concatenate((
    #     np.deg2rad(np.arange(-90.0, -80.0, angle_step)),
    #     np.deg2rad(np.arange(80.0, 90.0, angle_step))
    # ))
    
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def show_hough_line(img, accumulator, thetas, rhos, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')

    ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    # plt.axis('off')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def line_polar2cart(xmax, ymax, rho, theta):
    # p1 = [rho/np.cos(theta), 0]
    # p2 = [0, rho/np.sin(theta)]
    
    y0 = (rho - 0*np.cos(theta))/np.sin(theta)
    #y0 = ymax - y0
    y1 = (rho - xmax*np.cos(theta))/np.sin(theta)
    #y1 = ymax - y1
    
    p1 = [0, y0]
    p2 = [xmax, y1]
    
    return p1, p2
    
def filter_acc_ind(acc_ind):
    acc_ind_shift = np.vstack((np.array([-100, -100]).reshape(1, -1), acc_ind[:-1, :]))
    # print(acc_ind)
    # print(acc_ind_shift)
    ind_diff = acc_ind - acc_ind_shift
    
    acc_ind_diff_combine = np.hstack((acc_ind, ind_diff))
    #print(acc_ind_diff_combine)
    
    acc_ind_filter = []
    for r in acc_ind_diff_combine:
        #print(r)
        if (r[2] <= 10):
            if (r[3] <= 5):
                continue
                
        acc_ind_filter.append([r[0], r[1]])
        
    acc_ind_filter = np.array(acc_ind_filter)
    # print(acc_ind_filter)
    # #sys.exit(0)
    # print('---------------------')
    
    return acc_ind_filter
    
    
def main():
    view = 'endzone'
    blob_type = 'dog'
    blob_centre_folder = base_folder + '/images_processed/blob_centre/{0}/{1}'.format(view, blob_type)
    images_folder = base_folder + '/images_extract'
    
    img_files = os.listdir(blob_centre_folder)
    
    print(img_files)
    
    output_folder = base_folder + '/output/hash_marks/{0}'.format(view)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_img_folder = base_folder + '/images_processed/hash_marks/{0}'.format(view)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)
    
    cnt = 0
    for img_f in img_files:
        print(img_f)
        f_check = file_check(img_f, 'hash_marks', view)
        if (f_check):
            continue
        
        img = io.imread(blob_centre_folder + '/' + img_f)
        img_original = io.imread(images_folder + '/' + img_f)
        
        
        # if (view == 'endzone'):
        #     img = rotate(img, -90, resize = True)
        #     img_original = rotate(img_original, -90, resize = True)
            
        io.imshow(img)
        io.show()
        
        io.imshow(img_original)
        io.show()
        
        # io.imshow(img)
        # io.show()
        
        nr, nc = np.shape(img)
        #print(nr, nc)
        
        accumulator, thetas, rhos = hough_line(img)
        #show_hough_line(img, accumulator, thetas, rhos, save_path=None)
        
        if (view == 'sideline'):
            max_idx = np.argwhere(accumulator >= 0.8*np.max(accumulator))
        else:
            max_idx = np.argwhere(accumulator >= 0.5*np.max(accumulator))
        
        # idx = np.argmax(accumulator)
        # print(idx)
        # print(idx // accumulator.shape[1])
        # print(rhos)
        # print(np.shape(rhos))
        # print(accumulator)
        # print(np.shape(accumulator))
        # rho = rhos[idx // accumulator.shape[1]]
        # theta = thetas[idx % accumulator.shape[1]]
        # print("rho={0:.2f}, theta={1:.0f}".format(rho, np.rad2deg(theta)))
        
        img_h, img_w, _ = np.shape(img_original)
        
        max_idx_filter = filter_acc_ind(max_idx)
        line_points = []
        for rho_ind, theta_ind in max_idx_filter:
            rho = rhos[rho_ind]
            theta = thetas[theta_ind]
            
            print('dist and angle', rho, np.rad2deg(theta))
            
            p1, p2 = line_polar2cart(nc, nr, rho, theta)
            #print(p1, p2)
            xs, ys = [p1[0], p2[0]], [p1[1], p2[1]]
            print(xs, ys, img_h)
            
            if ((ys[0] < 0.1*img_h) or (ys[1] < 0.1*img_h)):
                continue
                
            if ((ys[0] > 0.9*img_h) or (ys[1] > 0.9*img_h)):
                continue
                
            line_points.append((np.array(xs), np.array(ys)))
        
        plot_lines(img, line_points, title = 'Lines')
        
        # plot_lines(img_original, line_points, title = 'Lines', save_path = output_img_folder + '/' + img_f)
        # 
        # fp = open(output_folder + '/' + img_f.split('.')[0] + '.txt', 'w')
        # for line in line_points:
        #     #print(line)
        #     
        #     l = [str(x) for x in list(line[0]) + list(line[1])]
        #     #print(l)
        #     
        #     fp.write(','.join(l) + '\n')
        #     
        # fp.close()
        
        # cnt += 1
        # if (cnt >= 10):
        #     break
        
        #sys.exit(0)


if __name__ == '__main__':
    # imgpath = 'imgs/binary_crosses.png'
    # img = imageio.imread(imgpath)
    # if img.ndim == 3:
    #     img = rgb2gray(img)
    # accumulator, thetas, rhos = hough_line(img)
    # show_hough_line(img, accumulator, save_path='imgs/output.png')
    
    main()
