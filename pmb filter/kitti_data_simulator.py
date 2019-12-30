import os
import argparse
import math
import cv2
import numpy as np
from config import kitti_label_dir, kitti_image_dir, kitti_oxts_dir, kitti_calib_dir, outputs_dir, roi_length, roi_width, roi_height
from kitti_coordinate_transforms import cam_to_imu_transform, oxts_0_frame_trans, cam_to_image_transform
import sys

parser = argparse.ArgumentParser(description='Kitti_tracking_directories')

parser.add_argument('--label_dir', type=str, default=kitti_label_dir)

parser.add_argument('--image_dir',type=str, default = kitti_image_dir)

args = parser.parse_args()

class  my_exception(Exception):
    '''Raise exception to exit from if clause'''
    pass

class simulator():

    def __init__(self,video_filename='0000'):
    
        self.frame_no = 0
        
        self.img_dir = os.path.join(args.image_dir, video_filename)
        
        self.label_path = os.path.join(args.label_dir,video_filename + '.txt')
        
        try:
            self.label_file = open(self.label_path,'r')
        except:
            self.label_file.close()
    
        self.video_path = os.path.join(args.image_dir,video_filename)
        
        oxts_path = os.path.join(kitti_oxts_dir, video_filename + '.txt')
        
        # transformation matrices to the world coordinates
        self.projs = oxts_0_frame_trans(oxts_path,int(video_filename))

        self.path_calib = os.path.join(kitti_calib_dir, video_filename + '.txt')
        
        self.cam_to_imu_proj = cam_to_imu_transform(self.path_calib)
        
        self.cam_to_img_proj = self.get_cam_to_image_proj()
        
        self.save_dir = os.path.join(outputs_dir, video_filename)

        if not(os.path.isdir(self.save_dir)):
            os.mkdir(self.save_dir)
        
    def get_cam_to_image_proj(self):
        '''
            Get 3X4 camera to image projection matrix.
        '''
        
        return cam_to_image_transform(self.path_calib)
        
        
    def get_imu_to_world_proj(self):
        '''
            Get the 4x4 projection matrix of imu to world
            coordinates before getting measurements.
        '''
        if (self.frame_no < len(self.projs)):
            return self.projs[self.frame_no]
        else:
            self.label_file.close()
            print('end of the file')
            sys.exit()
 
    def get_measurements(self):
        '''
            Get detections (measurements) from the current 
            image and then increase the frame no by one. 
        '''
    
        measurements = []
    
        if (self.frame_no == 0):
            self.line = self.label_file.readline()
        
        if (len(self.line) == 0):
            self.label_file.close()
            print('end of the file')
            sys.exit()
    
        parts = self.line.split()
        
        imu_to_world_proj = self.get_imu_to_world_proj()
        
        # parameters of the left camera projection matrix
        fx = self.cam_to_img_proj[0,0]
        fy = self.cam_to_img_proj[1,1]
        fxbx = self.cam_to_img_proj[0,-1]
        fyby = self.cam_to_img_proj[1,-1]
        ox = self.cam_to_img_proj[0,2]
        oy = self.cam_to_img_proj[1,2]
    
        while(int(parts[0]) == self.frame_no):
        
            object_class = parts[2]
            truncation = float(parts[3])
        
            try:
                if (truncation != 2 and object_class not in ['DontCare', 'Tram', 'Misc']):
                    
                    left, top, right, bottom = [float(p) for p in parts[6:10]] 
                    
                    # dimensions of 2D bbox
                    w = right - left
                    h = bottom - top
                    
                    # center of 2D bbox
                    mu_x = left + w/2 
                    mu_y = top + h/2
                    
                    depth = float(parts[-2]) - self.cam_to_img_proj[2,-1]
                    
                    # fxX + fxbx + ox*Z = mu_x*Z 
                    X = ( (mu_x - ox)*depth - fxbx ) / fx
                    # fyY + oy*Z  + fyby = mu_y*Z
                    Y = ( (mu_y - oy)*depth - fyby ) / fy
                    
                    if ( depth > roi_length or abs(X) > roi_width/2 or abs(Y) > roi_height ):
                        raise my_exception
                    
                    # transform x,y,z at the kth camera coordinates into the kth imu/gps coordinates
                    xyz1_imu = np.dot(self.cam_to_imu_proj, np.array([X,Y,depth,1]))
                    
                    # transform xyz1_imu to the world coordinates
                    xyz1_world = np.dot(imu_to_world_proj, xyz1_imu)
                    
                    # transfor homogenous coordinates to 3D world coordinates
                    xyz_world = xyz1_world[:-1]/xyz1_world[-1] 
                    
                    r = sum(map(lambda x: x**2, xyz_world))**0.5
                    
                    z = np.concatenate((xyz1_world[:-1],[r]))
                    
                    bbox_dims = np.array([w,h])
                    bbox_center = np.array([mu_x,mu_y])
        
                    measurements.append({ 'state': z,
                                          'object_class': object_class, 
                                          'cls_prob': 1., 
                                          'conf_prob': 1.,
                                          'bbox_dims': bbox_dims,
                                          'bbox_cntr': bbox_center
                                   })
        
            except my_exception:
                pass
            
            finally:
                self.line = self.label_file.readline()
            
            if (len(self.line) == 0):
                break
            
            parts = self.line.split()
        
        self.frame_no += 1
    
        return measurements

    def read_image(self):
        '''
            Read the current image before getting measurements.
        '''
        
        img_file = os.path.join(self.img_dir, '{:0>6}'.format(self.frame_no) + '.png')
        
        return cv2.imread(img_file)
        
        
    
        
    
   