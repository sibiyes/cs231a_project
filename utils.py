import os
import sys

from skimage import io

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def load_sideline_mask(img_f):
    sideline_mask_folder = base_folder + '/images_processed/sideview_mask'
    
    mask = io.imread(sideline_mask_folder + '/' + img_f, as_gray = True)
    mask = mask/255.0
    
    return mask
    
    
    
