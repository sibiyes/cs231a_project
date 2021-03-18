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

import matplotlib.pyplot as plt
from matplotlib import cm

from itertools import product

from gradient_analysis import gradient_analysis

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)



def rad2deg(angle_rad):
    angle_deg = (angle_rad * 180)/np.pi
    
    return angle_deg

def filter_by_proximity(line_points, axis):
    
    
    
    endpoints = []
    if (axis == 'x'):
        line_points = sorted(line_points, key = lambda x: x[0][0])
        endpoints = [c[0] for c in line_points]
    else:
        line_points = sorted(line_points, key = lambda x: x[1][0])
        endpoints = [c[1] for c in line_points]
    
    endpoints = np.array(endpoints)
    
    endpoints_shift = np.vstack((np.array([-10000, -10000]).reshape(1, -1), endpoints[:-1, :]))
    endpoints_diff = endpoints - endpoints_shift
    endpoints_diff_combine = np.hstack((endpoints, endpoints_diff))
    
    line_filter = []
    for r in endpoints_diff:
        if (np.abs(r[0]) <= 40):
            if (np.abs(r[1]) <= 40):
                line_filter.append(False)
                continue
                
        line_filter.append(True)
        
    
    line_points_filtered = []
    for i in range(len(line_filter)):
        if line_filter[i]:
            line_points_filtered.append(line_points[i])
    
    return line_points_filtered
    
    
def run_hough(image, rotate_img = False):
    
    
    #https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html
    
    # grad = gradient_analysis(image)
    # mask = color_filter(image)
    # io.imshow(mask)
    # io.show()
    #edges = canny(grad, 1.0)

    image = rgb2gray(image)
    img_h, img_w = np.shape(image)
    edges = canny(image, 1.0)
    
    
    
    #edges = roberts(image)
    # io.imshow(edges)
    # io.show()
    
    # edges_closed = binary_closing(edges)
    # io.imshow(edges_closed)
    # io.show()    
    # 
    # return
    
    image_original = np.copy(image)
    
    if (rotate_img == True):
        image = rotate(image, -90, resize = True)
        edges = rotate(edges, -90, resize = True)
        
    # print(np.shape(image))
    # return
    
    angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    #print(angles)
    h, theta, d = hough_line(edges, theta = angles)
    # print(h)
    # print(theta)
    # print(d)
    
    
    origin = np.array((0, image.shape[1]))
    
    line_points = []
    threshold = 0.5*np.max(h)
    #threshold = 0.01*np.max(h)
    #print('angle and dist')
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold = threshold)):
        #print(rad2deg(angle), dist)
        
        #print('angle and dist', angle, rad2deg(angle), dist)
        y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
        
        if ( (np.abs(y0) > 2*np.shape(image)[0]) or (np.abs(y1) > 2*np.shape(image)[0]) ):
            continue
            
        ### filter lines close to the edge of the image
        if ( (np.abs(y1-y0) <= 0.1) and ((np.abs(y1) < 3) or (np.abs(y1-image.shape[0]) < 3)) ):
            continue

        
        line_points.append((origin, np.array([y0, y1])))

    
    ### invert the rotation if rotated previously
    if (rotate_img == True):
        line_points_rotated = [(y, np.array([img_h, 0.0])) for x, y in line_points]
        line_points_rotated_filtered = []
        for xs, ys in line_points_rotated:
            angle = np.arctan((ys[1] - ys[0])/(xs[1] - xs[0]))
            angle = rad2deg(angle)
            print(angle) 
        
            if (np.abs(angle) > 20):
                line_points_rotated_filtered.append((xs, ys))
        
        image = image_original
        line_points = line_points_rotated_filtered
        
    
    return line_points
    
    
def detect_yard_lines(image, view, img_f):
    
    if (view == 'sideline'):
        rotate_img = True
        correction_axis = 'x'
    else:
        rotate_img = False
        correction_axis = 'y'
    
    line_points = run_hough(image, rotate_img)
    print(line_points)
    print(len(line_points))
    
    if (len(line_points) > 0):
        line_points = filter_by_proximity(line_points, correction_axis)
    
    #sys.exit(0)
    
    output_folder = base_folder + '/output/yardlines'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_img_folder = base_folder + '/images_processed/yardlines/{0}'.format(view)
    if not os.path.exists(output_img_folder):
        os.makedirs(output_img_folder)

    fp = open(output_folder + '/' + img_f.split('.')[0] + '.txt', 'w')
    for line in line_points:
        print(line)
        
        l = [str(x) for x in list(line[0]) + list(line[1])]
        print(l)
        
        fp.write(','.join(l) + '\n')
        
    fp.close()
    
    save_path = output_img_folder + '/' + img_f
    
    #plot_lines(image, line_points, 'Detected Lines')
    plot_lines(image, line_points, 'Detected Lines', save_path)
    
    
def detect_boundary(image, view, img_f):
    img_h, img_w, _ = np.shape(image)
    
    rotate_img = False
    line_points = run_hough(image, rotate_img)
    
    #plot_lines(image, line_points, 'Detected Lines')
    
    
    ### filter to keep only near horizontal lines
    line_points_filtered = []
    for xs, ys in line_points:
        angle = np.arctan((ys[1] - ys[0])/(xs[1] - xs[0]))
        angle = rad2deg(angle)
        print(angle)
        
        if (np.abs(angle) <= 15):
            line_points_filtered.append((xs, ys))
    
    #plot_lines(image, line_points_filtered, 'Detected Lines')
    
    
    upper_boundary_lines = []
    for xs, ys in line_points_filtered:
        if ((ys[0] < 0.3*img_h) or (ys[1] < 0.3*img_h)):
            upper_boundary_lines.append((xs, ys))
            
    lower_boundary_lines = []
    for xs, ys in line_points_filtered:
        if ((ys[0] > 0.7*img_h) or (ys[1] > 0.7*img_h)):
            lower_boundary_lines.append((xs, ys))
            
    # plot_lines(image, upper_boundary_lines, 'Upper Lines')
    # plot_lines(image, lower_boundary_lines, 'Lower Lines')
            
            
    upper_boundary_lines_left = sorted(upper_boundary_lines, key = lambda x: x[1][0], reverse = True)
    upper_boundary_lines_right = sorted(upper_boundary_lines, key = lambda x: x[1][1], reverse = True)
    # print('upper left')
    # print(upper_boundary_lines_left)
    # print('upper_right')
    # print(upper_boundary_lines_right)
    
    lower_boundary_lines_left = sorted(lower_boundary_lines, key = lambda x: x[1][0], reverse = False)
    
    line_upper = upper_boundary_lines_left[0:1]
    line_lower = lower_boundary_lines_left[0:1]
    
    
    # plot_lines(image, line_upper, 'Upper Boundary')
    # plot_lines(image, line_lower, 'Lower Boundary')
    
    output_folder = base_folder + '/output/boundary'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if (len(line_upper) + len(line_lower) != 0):
        fp = open(output_folder + '/' + img_f.split('.')[0] + '.txt', 'w')
        for line, position in [(line_upper, 'upper'), (line_lower, 'lower')]:
            if (len(line) == 0):
                continue
                
            line = line[0]
        
            l = [str(x) for x in list(line[0]) + list(line[1])]
            l.append(position)
            
            
            fp.write(','.join(l) + '\n')
            
        fp.close()
    
    return
    
    region_full = np.ones((img_h, img_w))
    
    if (len(line_upper) > 0):
        line = line_eq2(line_upper[0])
        region = get_line_regions(image, line)
        region[region == -1] = 0
        
        # io.imshow(region)
        # io.show()
        
        region_full = np.logical_and(region_full, region)
        
        
    if (len(line_lower) > 0):
        line = line_eq2(line_lower[0])
        region = get_line_regions(image, line)
        region[region == -1] = 0
        region = np.logical_not(region)
        
        # io.imshow(region)
        # io.show()
        
        region_full = np.logical_and(region_full, region)
    
    region_full = region_full.astype(int).astype(float)
    print(region_full)
    
    region_mask_folder = base_folder + '/images_processed/sideview_mask'
    if not os.path.exists(region_mask_folder):
        os.makedirs(region_mask_folder)
    
    #io.imshow(region_full)
    #io.show()
        
    # print('mask')
    # print(region_full)
    io.imsave(region_mask_folder + '/' + img_f, region_full)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    
    ax[0].imshow(image)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].imshow(region_full)
    ax[0].set_title('Region mask')
    ax[0].set_axis_off()
    
    plt.tight_layout()
    #plt.show()
    
    region_mask_comparison_folder = base_folder + '/images_processed/sideview_mask_comparison'
    if not os.path.exists(region_mask_comparison_folder):
        os.makedirs(region_mask_comparison_folder)
    
    plt.savefig(region_mask_comparison_folder + '/' + img_f)
    
def get_line_regions(image, line):
    img_h, img_w, _ = np.shape(image)
    
    indices = []
    for i in range(img_h):
        for j in range(img_w):
            indices.append([j, -1*i])
    
    #indices = np.array(list(product(np.arange(img_w), -1*np.arange(img_h))))
    indices = np.append(indices, np.ones(img_h*img_w).reshape(-1, 1), axis = 1)
    
    image_regions = np.sum(indices*line, axis = 1)
    image_regions = np.sign(image_regions)
    image_regions = image_regions.reshape((img_h, img_w))
    
    
    return image_regions
    
    
    
def line_eq(points):
    x1, x2 = points[0]
    y1, y2 = points[1]
    
    p1 = np.array((x1, y1, 1))
    p2 = np.array((x2, y2, 1))
    
    l = np.cross(p1, p2)

    print('line', l)
    
    print('validation')
    print('p1', np.dot(p1, l))
    print('p2', np.dot(p2, l))
    
    return l
    
def line_eq2(points):
    x1, x2 = points[0]
    y1, y2 = -1*points[1]
    
    # print('points')
    # print(x1, y1)
    # print(x2, y2)
    
    m1 = (y2 - y1)/(x2 - x1)
    c1 = y1 - m1*x1
    
    l = np.array([m1, -1, c1])
    
    p1 = np.array((x1, y1, 1))
    p2 = np.array((x2, y2, 1))
    
    # print('line', l)
    # 
    # print('validation')
    # print('p1', np.dot(p1, l))
    # print('p2', np.dot(p2, l))
    
    return l
    
    
def plot_lines(image, line_points, title = 'Lines', save_path = None):
    print(line_points)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()
    
    ax[1].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    
    for xs, ys in line_points:
        print(xs)
        print(ys)
        
        ax[1].plot(xs, ys, '-r')
        
    ax[1].set_xlim(origin)
    ax[1].set_ylim((image.shape[0], 0))
    ax[1].set_axis_off()
    ax[1].set_title(title)
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
    
    
