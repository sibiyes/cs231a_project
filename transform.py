import os
import sys

import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)


def plot_points(image, points, title = 'Points', save_path = None):
    
    plt.imshow(image)
    origin = np.array((0, image.shape[1]))
    
    for x, y in points:
        plt.plot(x, y, marker = 'o', color = 'red')
        
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
    plt.close()


def load_point_corres(point_f):
    fp = open(point_f, 'r')
    line = fp.readline()
    points = []
    while (line != ''):
        p_vals = [float(x) for x in line.strip().split(',')]
        points.append(p_vals)
        line = fp.readline()
        
    fp.close()
    
    points = np.array(points)
    
    return points
    
def load_person_loc(point_f):
    fp = open(point_f, 'r')
    line = fp.readline()
    points = []
    while (line != ''):
        p_vals = [int(x) for x in line.strip().split(',')]
        points.append(p_vals)
        line = fp.readline()
        
    fp.close()
    
    points = np.array(points)
    
    return points
    
def fit_homography(points):    
    A = []
    
    for p in points:
        x1 = p[0]
        y1 = p[1]
        x2 = p[2]
        y2 = p[3]
        
        a1 = [x1, y1, 1, 0, 0, 0, -1*x1*x2, -1*y1*x2, -1*x2]
        a2 = [0, 0, 0, x1, y1, 1, -1*x1*y2, -1*y1*y2, -1*y2]
        
        A.append(a1)
        A.append(a2)
        
    A = np.array(A)
    
    # print(A)
    # print(np.shape(A))
    
    u, s, v = np.linalg.svd(A)
    
    h = v[-1]
    
    #print(h)
    
    H = h.reshape(3, 3)
    
    return H
    
def template_img(pw = 4, pad_h = 30, pad_w = 30):    
    fh = 160
    fw = 300
    img_h = int(pw*(fh + 2*pad_h))
    img_w = int(pw*(fw + 2*pad_w))
    
    img = np.zeros((img_h, img_w, 3))
    img[:, :, 1] = 255.0
    
    ### draw yardlines
    yard_loc = np.arange(0, 110*3, 10*3)
    
    for yl in yard_loc:
        img_yl = pw*(pad_w + yl)
        
        hu = int(pw*pad_h)
        hl = int(pw*(pad_h+fh))
        wl = int(img_yl-2)
        wr = int(img_yl+2)
        
        img[hu:hl, wl:wr, :] = np.array([255, 255, 255])
    
    ### draw boundary and hash_marks
    
    bh_loc = [0, 69.9, 90.5, 160]
    for bhl in bh_loc:
        img_bhl = pw*(pad_h+bhl)
        
        hu = int(img_bhl-2)
        hl = int(img_bhl+2)
        wl = int(pw*pad_w) 
        wr = int(pw*(pad_w+fw))
        
        img[hu:hl, wl:wr, :] = np.array([255, 255, 255])
        
    # io.imshow(img)
    # io.show()
    
    return img
    
def translate_loc(p_loc, pw = 4, pad_h = 30, pad_w = 30):
    p_loc[:, 0] += pad_w
    p_loc[:, 1] += pad_h
    
    p_loc *= pw
    
    p_loc = p_loc.astype(int)
    
    return p_loc
    
def point_transform(points, H):
    points = np.hstack((points, np.ones(len(points)).reshape(-1, 1)))
    
    points_transform = (H @ points.T).T
    points_transform = points_transform/points_transform[:, 2].reshape(-1, 1)
    points_transform = points_transform[:, :2]
    
    return points_transform
    
    
    
def main():
    points_corres_folder = base_folder + '/output/points_corres'
    person_location_folder = base_folder + '/output/person_location'
    
    points_files = os.listdir(points_corres_folder)
    #print(points_files)
    
    #sys.exit(0)
    
    transform_image_output_older = base_folder + '/images_processed/image_transformed'
    if not os.path.exists(transform_image_output_older):
        os.makedirs(transform_image_output_older)
    
    for points_f in points_files:
        print(points_f)
        point_corres = load_point_corres(points_corres_folder + '/' + points_f)
        person_loc = load_person_loc(person_location_folder + '/' + points_f)
        #print(point_corres)
        #print(person_loc)
        
        point_loc = translate_loc(point_corres[:, 2:])
        #print(point_loc)
        
        point_corres[:, 2:] = point_loc
        
        #print(point_corres)
        
        H = fit_homography(point_corres)
        #print(H)
        
        img = template_img()
        
        # io.imshow(img)
        # io.show()
        
        point_loc_transform = point_transform(point_corres[:, :2], H)
        point_loc_transform = point_loc_transform.astype(int)
        print(point_loc_transform)
        
        person_loc_transform = point_transform(person_loc, H)
        person_loc_transform = person_loc_transform[person_loc_transform[:, 1] > 120] 
        
        
        save_path = transform_image_output_older + '/' + points_f.replace('.txt', '.jpg')
        
        # plot_points(img, point_loc, title = 'Points', save_path = None)
        # plot_points(img, point_loc_transform, title = 'Points', save_path = None)
        #plot_points(img, person_loc_transform, title = 'Points', save_path = None)
        plot_points(img, person_loc_transform, title = 'Points', save_path = save_path)
        
    
if __name__ == '__main__':
    main()
