import os
import sys
import numpy as np

from skimage import io
from skimage.color import rgb2gray, rgb2hsv

import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

from plot_lines_all import load_lines
from line_detection import line_eq2


### https://sportsknowhow.com/football/field-dimensions/nfl-football-field-dimensions.html

"""
57586_000540_sideline_70
57586_000540_sideline_150
57596_002686_sideline_170
57597_000658_sideline_10
57684_001985_sideline_50
57775_000933_sideline_70
57686_002546_sideline_170

"""

def get_yardmarks(img_f):
    yardline_marking_folder = base_folder + '/output/yardline_marking'
    yardline_marking_file = yardline_marking_folder + '/' + img_f.replace('.jpg', '.txt')
    
    fp = open(yardline_marking_file, 'r')
    line = fp.readline()
    fp.close()
    
    #markings = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    markings = line.strip().split(',')
    markings = [int(m) for m in markings]
    
    return markings
    
def point_int(ln1, ln2):
    p = np.cross(ln1, ln2)
    p = p/p[2]
    p = p[:2]
    
    return p
    
def plot_points(img_f, points, title = 'Points', save_path = None):
    
    images_folder = base_folder + '/images_extract'
    image = io.imread(images_folder + '/' + img_f)

    plt.imshow(image)
    
    origin = np.array((0, image.shape[1]))
    
    for x, y in points:
        #print(x, y)
        plt.plot(x, -1*y, marker = 'o', color = 'red')
    
    plt.tight_layout()
    
    if (save_path == None):
        plt.show()
    else:
        plt.savefig(save_path)
        
### endzone = 1 right
### endzone = -1 left
endzone_view = {
    '57586_000540_endzone_70.jpg': 1,
    '57586_000540_endzone_150.jpg': 1,
    '57596_002686_endzone_170.jpg': -1,
    '57597_000658_endzone_10.jpg': -1,
    '57684_001985_endzone_50.jpg': -1,
    '57775_000933_endzone_70.jpg': 1,
    '57686_002546_endzone_170.jpg': 1
}
    

def main():
    # img_files = [
    #     '57686_002546_sideline_170.jpg'
    # ]
    
    # img_files = [
    #     '57586_000540_sideline_70.jpg',
    #     '57586_000540_sideline_150.jpg',
    #     '57596_002686_sideline_170.jpg',
    #     '57597_000658_sideline_10.jpg',
    #     '57684_001985_sideline_50.jpg',
    #     '57775_000933_sideline_70.jpg',
    #     '57686_002546_sideline_170.jpg'
    # ]
    
    # img_files = [
    #     '57686_002546_endzone_170.jpg'
    # ]
    
    img_files = [
        '57586_000540_endzone_70.jpg',
        '57586_000540_endzone_150.jpg',
        '57596_002686_endzone_170.jpg',
        '57597_000658_endzone_10.jpg',
        '57684_001985_endzone_50.jpg',
        '57775_000933_endzone_70.jpg',
        '57686_002546_endzone_170.jpg'
    ]
    
    view = 'endzone'
    
    for img_f in img_files:
        lines_all  = load_lines(img_f, view)
        #print(lines_all)
        
        yardlines = lines_all['yardlines']
        boundary = lines_all['boundary']
        hash_marks = lines_all['hash_marks']
        
        #print(yardlines)
        
        
        if (view == 'sideline'):
            yardlines = sorted(yardlines, key = lambda x: x[0][0])
        else:
            yardlines = sorted(yardlines, key = lambda x: x[1][0])
            #print(yardlines)
            
            
        #continue
        
        yard_markings = get_yardmarks(img_f)
        
        
        if (view == 'sideline'):
            hash_marks = sorted(hash_marks, key = lambda x: x[1][0])
            hash_marks = dict(zip(['upper', 'lower'], hash_marks))
        else:
            hash_marks = sorted(hash_marks, key = lambda x: x[0][0])
            ez = endzone_view[img_f]
            if (ez == 1):
                hash_marks = dict(zip(['lower', 'upper'], hash_marks))
            else:
                hash_marks = dict(zip(['upper', 'lower'], hash_marks))
        
        
        yardlines_markings = []

        for i in range(len(yardlines)):
            yardlines_markings.append((yardlines[i][0], yardlines[i][1], yard_markings[i]))
        
        hu_ln = line_eq2(hash_marks['upper'])
        hl_ln = line_eq2(hash_marks['lower'])
        
        bu_ln = []
        bl_ln = []
        if (boundary.get('upper') != None):
            bu_ln = line_eq2(boundary['upper'][0])
        if (boundary.get('lower') != None):
            bl_ln = line_eq2(boundary['lower'][0])

        point_correspondence = []
        for l in yardlines_markings:
            xs = l[0]
            ys = l[1]
            mark = l[2]
            
            yard_ln = line_eq2([xs, ys])
            print('yard line', yard_ln)
            p_int_hu = point_int(yard_ln, hu_ln)
            p_int_hu_coord = [mark*3, 69.9]
            p_int_hl = point_int(yard_ln, hl_ln)
            p_int_hl_coord = [mark*3, 90.5]
            
            
            # print('int hu', p_int_hu)
            # print('int hl', p_int_hl)
            
            point_correspondence.append((p_int_hu, p_int_hu_coord))
            point_correspondence.append((p_int_hl, p_int_hl_coord))
            
            if (len(bu_ln) > 0):
                p_int_bu = point_int(yard_ln, bu_ln)
                p_int_bu_coord = [mark*3, 0]
                point_correspondence.append((p_int_bu, p_int_bu_coord))
            
        print(point_correspondence)
        
        points = [x[0] for x in point_correspondence]
        plot_points(img_f, points, title = 'Points', save_path = None)
        
        points_corres_output_folder = base_folder + '/output/points_corres'
        if not os.path.exists(points_corres_output_folder):
            os.makedirs(points_corres_output_folder)
            
        fp = open(points_corres_output_folder + '/'  + img_f.replace('.jpg', '.txt'), 'w')
        for p in point_correspondence:
            print(p)
            p1 = p[0]
            p2 = p[1]
            p_output = [str(p1[0]), str(p1[1]), str(p2[0]), str(p2[1])]
            print(p_output)
            
            fp.write(','.join(p_output) + '\n')
            
        fp.close()
            
        
    
if __name__ == '__main__':
    main()
