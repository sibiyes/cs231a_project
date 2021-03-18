import os
import sys

import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def load_yolo_output(output_file):
    fp = open(output_file, 'r')
    line = fp.readline()
    
    bounding_box = []
    while (line != ''):
        line = line.strip()
        
        if ('person' not in line):
            line = fp.readline()
            continue
            
        line = ' '.join(line.split())
        line = line.replace('(', '').replace(')', '').replace(':', '')
        line_elem = line.split()
        
        conf = float(line_elem[1].replace('%', ''))
        left_x = int(line_elem[3])
        top_y = int(line_elem[5])
        width = int(line_elem[7])
        height = int(line_elem[9])
        
        if (width < 400):
            bounding_box.append([left_x, top_y, width, height, conf])
        
        #print(conf, left_x, top_y, width, height)
        
        line = fp.readline()
        
    bounding_box = np.array(bounding_box)
    #print(bounding_box)
        
    fp.close()
    
    return bounding_box
    
def get_location(bounding_box):
    loc = []
    for box in bounding_box:
        print(box)
        x = int(box[0] + box[2]/2)
        y = int(box[1] + box[3])
        
        loc.append([x, y])
    
    loc = np.array(loc)
    
    return loc
    
def plot_points(img_f, points, title = 'Points', save_path = None):
    
    images_folder = base_folder + '/images_extract'
    #images_folder = base_folder + '/images_processed/yolo_processed'
    image = io.imread(images_folder + '/' + img_f)

    #fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    
    
    plt.imshow(image)
    #plt.set_title('Input image')
    #plt.set_axis_off()
    
    # ax[1].imshow(image, cmap=cm.gray)
    origin = np.array((0, image.shape[1]))
    
    for x, y in points:
        #print(x, y)
        plt.plot(x, -1*y, marker = 'o', color = 'red')
        
    #plt.set_xlim(origin)
    #plt.set_ylim((image.shape[0], 0))
    #plt.set_axis_off()
    #plt.set_title(title)
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
    plt.close()
        

def main():
    yolo_output_folder = base_folder + '/output/yolo_output'
    person_location_folder = base_folder + '/output/person_location'
    person_location_image_folder = base_folder + '/images_processed/person_location'
    
    if not os.path.exists(person_location_folder):
        os.makedirs(person_location_folder)

    if not os.path.exists(person_location_image_folder):
        os.makedirs(person_location_image_folder)

    
    for output_f in os.listdir(yolo_output_folder):
        print(output_f)
        bounding_box = load_yolo_output(yolo_output_folder + '/' + output_f)
        print(bounding_box)
        
        points = get_location(bounding_box)
        points[:, 1] *= -1
        
        img_f = output_f.replace('.txt', '.jpg')
        
        img_save_path = person_location_image_folder + '/' + img_f
        #plot_points(img_f, points, title = 'Points')
        plot_points(img_f, points, title = 'Points', save_path = img_save_path)
        
        fp = open(person_location_folder + '/' + output_f, 'w')
        for p in points:
            fp.write(','.join([str(x) for x in p]) + '\n')
            
        fp.close()
        
        #sys.exit(0)
    
if __name__ == '__main__':
    main()
