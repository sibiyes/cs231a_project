import numpy as np
from itertools import product

from line_detection import line_eq, line_eq2

def main():
    # x1, y1 = 0, -139.63255782785694
    # x2, y2 = 1280, -94.93397287841971
    
    x1, y1 = 0, 139.63255782785694
    x2, y2 = 1280, 94.93397287841971
    
    l = line_eq2(np.array([(x1, x2), (y1, y2)]))
    
    s = m1 = (y2 - y1)/(x2 - x1)
    print('s', s)
    
    print(l)
    
    print('slope', -1*(l[0]/l[1]))
    print('intercept', -1*(l[2]/l[1]))
    
    print('test')
    print(np.dot(l, np.array([0, 0, 1])))
    print(np.dot(l, np.array([1280, 0, 1])))
    print(np.dot(l, np.array([0, -720, 1])))
    print(np.dot(l, np.array([1280, -720, 1])))
    
    print('-----------------------')
    get_line_regions(l)
    

def get_line_regions(line):
    #img_h, img_w, _ = np.shape(image)
    img_h, img_w = 720, 1280
    
    indices = []
    for i in range(img_h):
        for j in range(img_w):
            indices.append([j, -1*i])
    
    #indices = np.array(list(product(np.arange(img_w), -1*np.arange(img_h))))
    indices = np.append(indices, np.ones(img_h*img_w).reshape(-1, 1), axis = 1)
    
    image_segments = np.sum(indices*line, axis = 1)
    image_segments = np.sign(image_segments)
    image_segments = image_segments.reshape((img_h, img_w))
    
    
    



    
    
    
if __name__ == '__main__':
    main()
