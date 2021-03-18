import os
import sys

import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv

import matplotlib.pyplot as plt
from matplotlib import cm


script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

from analysis import get_img_files, file_check


def load_lines(img_f, view):
    yardlines_folder = base_folder + '/output/yardlines'
    boundary_folder = base_folder + '/output/boundary'
    hash_marks_folder = base_folder + '/output/hash_marks/{0}'.format(view)
    
    yardlines = []
    fp = open(yardlines_folder + '/' + img_f.replace('.jpg', '.txt'))
    line = fp.readline()
    while (line != ''):
        line_elem = line.strip().split(',')
        xs = np.array([float(line_elem[0]), float(line_elem[1])])
        ys = np.array([float(line_elem[2]), float(line_elem[3])])
        
        
        yardlines.append((xs, ys))
        
        line = fp.readline()
    
    boundary = {}
    boundary_file = boundary_folder + '/' + img_f.replace('.jpg', '.txt')
    if (os.path.exists(boundary_file)):
        fp = open(boundary_file, 'r')
        line = fp.readline()
        while (line != ''):
            line_elem = line.strip().split(',')
            xs = np.array([float(line_elem[0]), float(line_elem[1])])
            ys = np.array([float(line_elem[2]), float(line_elem[3])])
            
            b = line_elem[4]
            boundary[b] = [(xs, ys)]
            
            line = fp.readline()
    
    hash_marks = []
    fp = open(hash_marks_folder + '/' + img_f.replace('.jpg', '.txt'))
    line = fp.readline()
    while (line != ''):
        line_elem = line.strip().split(',')
        xs = np.array([float(line_elem[0]), float(line_elem[1])])
        ys = np.array([float(line_elem[2]), float(line_elem[3])])
        
        hash_marks.append((xs, ys))
        
        line = fp.readline()
    
    lines_all = {
        'yardlines': yardlines,
        'boundary': boundary,
        'hash_marks': hash_marks
    }
    
    #print(lines_all)
    
    return lines_all
    
def plot_lines(image, line_points, title = 'Lines', save_path = None):

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    
    for xs, ys in line_points['yardlines']:
        ax[1].plot(xs, ys, '-r')
        
    if (line_points['boundary'].get('upper') != None):
        for xs, ys in line_points['boundary']['upper']:
            ax[1].plot(xs, ys, '-b')
            
    if (line_points['boundary'].get('lower') != None):
        for xs, ys in line_points['boundary']['lower']:
            ax[1].plot(xs, ys, '-b')
            
    for xs, ys in line_points['hash_marks']:
        ax[1].plot(xs, ys, '-y')
        
    
        
    ax[1].set_xlim(origin)
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title(title)
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)

def main():
    view = 'endzone'
    
    images_folder, image_files = get_img_files()
    
    if (view == 'endzone'):
        image_files = [
                '57586_000540_endzone_70.jpg',
                '57586_000540_endzone_150.jpg',
                '57596_002686_endzone_170.jpg',
                '57597_000658_endzone_10.jpg',
                '57684_001985_endzone_50.jpg',
                '57775_000933_endzone_70.jpg',
                '57686_002546_endzone_170.jpg'
        ]
    
    cnt = 0
    
    image_lines_folder = base_folder + '/images_processed/lines_all'
    
    if (not os.path.exists(image_lines_folder)):
        os.makedirs(image_lines_folder)
    
    for img_f in image_files:
        print(img_f)
        #view = None
        if ('endzone' in img_f):
            view = 'endzone'
        else:
            view = 'sideline'
            
        # if (view == 'endzone'):
        #     continue

        f_check = file_check(img_f, 'lines_all', view)
        
        if (f_check):
            print('f check')
            continue
            
        image = io.imread(images_folder + '/' + img_f)
        image_gray = rgb2gray(image)
        
        # io.imshow(image)
        # io.show()
        # 
        # continue
        
        lines_all  = load_lines(img_f, view)
        
        # print(lines_all)
        # continue
        
        #plot_lines(image, lines_all)
        plot_lines(image, lines_all, save_path = image_lines_folder + '/' + img_f)
        
        #sys.exit(0)
    
        # cnt += 1
        # if (cnt >= 10):
        #     break
    
    
if __name__ == '__main__':
    main()
