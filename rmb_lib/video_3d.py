import random
import os
import numpy as np
from PIL import Image
from rmb_lib.data_augment import transform_data


class Video_3D:
    def __init__(self, info_list, tag, stream, img_format='frame{:06d}.jpg'):
        '''
            info_list: [name, path, total_frame, label]
            tag: 'rgb'(default) or 'flow'
            img_format: 'frame{:06d}{}.jpg'(default)
        '''
        self.name = info_list[0]
        self.path = info_list[1]
        
        if isinstance(info_list[2], int):
            self.total_frame_num = info_list[2]
        else:
            self.total_frame_num = int(info_list[2])
        
        if isinstance(info_list[3], int):
            self.label = info_list[3]
        else:
            self.label = int(info_list[3])
        '''
        if isinstance(info_list[3], int):
            self.start_frame = info_list[3]
        else:
            self.start_frame = int(info_list[3])
        if isinstance(info_list[4], int):
            self.label = info_list[4]
        else:
            self.label = int(info_list[4])
        '''  
      
        self.tag = tag
        self.stream = stream
        
        if self.stream == 'pose':
            self.img_format = 'frame{:06d}.png'
        else:   
            #self.img_format = self.name + '_{}.jpg'
            self.img_format = img_format
    
          
    def get_frames(self, frame_num, side_length=224, is_numpy=True, data_augment=False):
        '''
        frame_num : clip size
        start_frame 
        '''
        #assert frame_num <= self.total_frame_num
        frames = list()
        start = random.randint(1, max(self.total_frame_num-frame_num, 0)+1)
        #start = self.start_frame
        for i in range(start, start+frame_num):
            frames.extend(self.load_img((i-1)%self.total_frame_num+1))
            #frames.extend(self.load_img(i+1))

        frames = transform_data(frames, crop_size=side_length, random_crop=data_augment, random_flip=data_augment)

        if is_numpy:
            frames_np = []
            if self.tag == 'rgb':
                for i, img in enumerate(frames):
                    frames_np.append(np.asarray(img))
            elif self.tag == 'flow':
                for i in range(0, len(frames), 2):
                    tmp = np.stack([np.asarray(frames[i]), np.asarray(frames[i+1])], axis=2)
                    frames_np.append(tmp)
            return np.stack(frames_np)

        return  frames
    

    def load_img(self, index):
        img_dir = self.path
        if self.tag == 'rgb':
            if 'pair' in self.stream:
                p_img = Image.open(os.path.join(img_dir.format('p'), self.img_format.format(index, '')))
                o_img = Image.open(os.path.join(img_dir.format('o'), self.img_format.format(index, '')))

                image_data = (0.5*np.array(p_img) + 0.5*np.array(o_img)).astype(np.uint8)
                return [Image.fromarray(image_data)]
            else:
                return [Image.open(os.path.join(img_dir, self.img_format.format(index))).convert('RGB')]
        if self.tag == 'flow':
            if 'pair' in self.stream:
                p_img = Image.open(os.path.join(img_dir.format('p'), self.img_format.format(index, ''))).convert('L')
                o_img = Image.open(os.path.join(img_dir.format('o'), self.img_format.format(index, ''))).convert('L')
                return [p_img,o_img]
            else: 
                # u_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_u'))).convert('L')
                # v_img = Image.open(os.path.join(img_dir, self.img_format.format(index, '_v'))).convert('L')
                u_img = Image.open(os.path.join(img_dir.format('u'), self.img_format.format(index, ''))).convert('L')
                v_img = Image.open(os.path.join(img_dir.format('v'), self.img_format.format(index, ''))).convert('L')
                return [u_img,v_img]
            
        return

    def __str__(self):
        return 'Video_3D:\nname: {:s}\nframes: {:d}\nlabel: {:d}\nPath: {:s}'.format(
            self.name, self.total_frame_num, self.label, self.path)
