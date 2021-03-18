import os
import sys

import numpy as np
import random
from collections import defaultdict

from skimage import io
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import match_template
from skimage.transform import rotate
from skimage.feature import peak_local_max 

import matplotlib.pyplot as plt

from analysis import get_img_files, file_check

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def plot_points(image, points, title = 'Points', save_path = None):
    plt.imshow(image)
    
    origin = np.array((0, image.shape[1]))
    
    for x, y in points:
        #print(x, y)
        plt.plot(y, x, marker = 'o', color = 'red')
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
    plt.close()
        
def plot_points2(image, points, title = 'Points', save_path = None):
    plt.imshow(image)
    
    origin = np.array((0, image.shape[1]))
    
    ax = plt.subplot(1, 1, 1)
    
    ax.imshow(image, cmap=plt.cm.gray)
    ax.set_axis_off()
    ax.set_title('image')
    # highlight matched region
    
    for x, y in points:
        #print(x, y)
        #plt.plot(y, x, marker = 'o', color = 'red')
        rect = plt.Rectangle((y, x), 200, 200, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.plot(y, x, marker='v', color="green")
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
    plt.close()

def get_template_samples():
    template_folder = base_folder + '/images_processed/pattern_match_template/sideline_test_gray'
    print(template_folder)
    
    image_files_sample = []
    for sub_folder in os.listdir(template_folder):
        #print(sub_folder)
        img_files = os.listdir(template_folder + '/' + sub_folder)
        img_f = random.sample(img_files, 1)
        img_f_path = template_folder + '/' + sub_folder + '/' + img_f[0]
        image_files_sample.append(img_f_path)
        
    #print(image_files_sample)
    
    return image_files_sample

def match_sub(image, image_sub):
    print('shape', np.shape(image))
    result = match_template(image, image_sub)
    print(result)
    print(np.shape(result))
    print('max', np.max(result))
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]
    print('xy', x, y)

    fig = plt.figure(figsize=(12, 8))
    #fig = plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    #ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(image_sub, cmap=plt.cm.gray)
    ax1.set_axis_off()
    ax1.set_title('template')

    ax2.imshow(image, cmap=plt.cm.gray)
    ax2.set_axis_off()
    ax2.set_title('image')
    # highlight matched region
    h_sub, w_sub = image_sub.shape
    print('hw', h_sub, w_sub)
    rect = plt.Rectangle((x, y), w_sub, h_sub, edgecolor='r', facecolor='none')
    ax2.add_patch(rect)
    ax2.plot(x, y, marker='v', color="green")

    ax3.imshow(result)
    ax3.set_axis_off()
    ax3.set_title('`match_template`\nresult')
    # highlight matched region
    ax3.autoscale(False)
    #ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

    plt.show()
    
def match_sub_multiple(image, image_sub):
    result = match_template(image, image_sub)
    
    peaks = peak_local_max(result, min_distance=50, threshold_rel = 0.85)
    
    h_sub, w_sub = image_sub.shape
    
    peaks[:, 0] += h_sub//2
    peaks[:, 1] += w_sub//2
    
    return peaks
    
    

def main():
    
    pattern_match_folder = base_folder + '/images_processed/pattern_match_test'
    full_image_file = pattern_match_folder + '/full_image/57913_000218_sideline_90.jpg'
    
    sub_images_folder = pattern_match_folder + '/sub_image'
    sub_image_files = os.listdir(sub_images_folder)
    
    sub_image_files = get_template_samples()
    
    images_folder, image_files = get_img_files()
    
    
    yardmark_detection_output_folder = base_folder + '/images_processed/yardmark_detection'
    yardmark_pattern_match_output_folder = base_folder + '/images_processed/yardmark_pattern_match'
    
    if (not os.path.exists(yardmark_detection_output_folder)):
        os.makedirs(yardmark_detection_output_folder)
        
    if (not os.path.exists(yardmark_pattern_match_output_folder)):
        os.makedirs(yardmark_pattern_match_output_folder)
    
    for img_f in image_files:
        print(img_f)
        img_f = pattern_match_folder + '/full_image/57913_000218_sideline_90.jpg'
        if ('endzone' in img_f):
            view = 'endzone'
        else:
            view = 'sideline'
            
        if (view == 'endzone'):
            continue

        #f_check = file_check(img_f, 'lines_all', view)
            
        #image_full = io.imread(images_folder + '/' + img_f)
        image_full = io.imread(img_f)
        image_full = rgb2gray(image_full)
        
        # image_full = io.imread(full_image_file)
        # image_full = rgb2gray(image_full)
        
        # io.imshow(image_full)
        # io.show()
        
        peaks_all = []
        for img_sub_f in sub_image_files:
            #image_sub = io.imread(sub_images_folder + '/' + img_sub_f)
            image_sub = io.imread(img_sub_f)
            image_sub = rgb2gray(image_sub)
            #image_sub_rotate = rotate(image_sub, 180)
            
            # io.imshow(image_sub)
            # io.show()
            
            # io.imshow(image_sub_rotate)
            # io.show()
            
            #peaks = match_sub_multiple(image_full, image_sub)
            peaks = match_sub(image_full, image_sub)
            
            continue
            
            peaks_all.append(peaks)
            
        sys.exit(0)
            
        peaks_all = np.concatenate(peaks_all, axis = 0)
        #print(peaks_all)
            
        #plot_points(image_full, peaks_all)
        #sys.exit(0)
        
        cluster_points = defaultdict(list)
        points = peaks_all.copy()
        
        c = 1
        while len(points) > 0:
            p0 = points[0]
            cluster_points[c].append(p0)
            
            points_new = []
            #print('p0', p0)
            for p in points[1:]:
                #print(p)
                d = np.linalg.norm(p-p0)
                
                if (d < 80):
                    #print('*')
                    cluster_points[c].append(p)
                else:
                    points_new.append(list(p))
                    
            # print(np.array(points_new))
            # print(cluster_points)
            
            c += 1
            points = np.array(points_new)
                
            
        #print(cluster_points)
        
        cluster_mean = {}

        for c, points in cluster_points.items():
            points = np.concatenate([p.reshape(1, -1) for p in points], axis = 0)
            center = np.mean(points, axis = 0)
        
            cluster_mean[c] = center.astype(int)
            #plot_points(image_full, points)

        #print(cluster_mean)
        
        points_all = []
        for c, points in cluster_mean.items():
            
            points_all += [points]
        
        save_path = yardmark_detection_output_folder + '/' + img_f
        plot_points(image_full, points_all, save_path = save_path)
        
        #plot_points(image_full, points_all)
        
        #sys.exit(0)
    
if __name__ == '__main__':
    main()
