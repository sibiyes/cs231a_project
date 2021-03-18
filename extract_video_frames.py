import os
import sys
import numpy as np
from collections import defaultdict

from skimage import io
import imageio

script_folder = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.dirname(script_folder)

def get_video_files():
    video_folder = base_folder + '/nfl-impact-detection/train'
    video_files = os.listdir(video_folder)
    
    video_file_map = defaultdict(dict)
    for f in video_files:
        video_tag = '_'.join(f.split('_')[:2])
        view = f.split('.')[0].split('_')[-1]
        video_file_map[video_tag][view] = f
        
    return video_folder, video_file_map
    


def main():
    video_folder, video_files = get_video_files()
    
    video_output_folder = base_folder + '/images_extract'
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
        
    for video_tag in video_files:
        video_sideline_file = video_files[video_tag]['Sideline']
        video_endzone_file = video_files[video_tag]['Endzone']
        
        print(video_sideline_file)
        print(video_endzone_file)
        
        video_sideline = imageio.get_reader(video_folder + '/' + video_sideline_file, 'ffmpeg')
        video_endzone = imageio.get_reader(video_folder + '/' + video_endzone_file, 'ffmpeg')
        print(video_sideline)
        
        frames = np.arange(10, 210, 20)    
        
        for frame_num in frames:
            frame_img_sideline = video_sideline.get_data(frame_num)
            frame_img_endzone = video_endzone.get_data(frame_num)
            
            # io.imshow(frame_img_sideline)
            # io.show()
            # 
            # io.imshow(frame_img_endzone)
            # io.show()
            
            io.imsave(video_output_folder + '/{0}_{1}_{2}.jpg'.format(video_tag, 'sideline', frame_num) , frame_img_sideline)
            io.imsave(video_output_folder + '/{0}_{1}_{2}.jpg'.format(video_tag, 'endzone', frame_num) , frame_img_endzone)
        
        #sys.exit(0)
    
    # filename = '/tmp/file.mp4'
    # vid = imageio.get_reader(filename,  'ffmpeg')
    # nums = [10, 287]
    # for num in nums:
    #     image = vid.get_data(num)
    #     fig = pylab.figure()
    #     fig.suptitle('image #{}'.format(num), fontsize=20)
    #     pylab.imshow(image)
    # pylab.show()
    
if __name__ == '__main__':
    main()
