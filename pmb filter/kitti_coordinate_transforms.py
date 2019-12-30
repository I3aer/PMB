import os
import functools
import numpy as np
import sys
import matplotlib.pyplot as plt

from config import kitti_oxts_dir

def cam_to_imu_transform(calib_path):
    '''
        This method returns transformation matrix to convert 3D camera coordinates to GPS/IMU coordinates.

        To transform a point X from GPS/IMU coordinates to the 3D camera coordinates:
                    Y =  (R|T)_velo_to_cam * (R|T)_imu_to_velo * X,
        where
            - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
            - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates

        Inputs: calib_path is the path of the calibration file.
        Output: 4x4 transformation matrix from 3D camera coordinates to GPS/IMU
                coordinates.
    '''

    # transform a point X from GPS/IMU coordinates to the camera coordinates
    Tr_imu_cam = imu_to_cam_transform(calib_path)

    # return cam to imu/gps transformation matrix
    return np.linalg.inv(Tr_imu_cam)

def cam_to_image_transform(calib_path):
    '''
        To transform a point X from left camera coordinates to the pixel p=(u,v) coordinates:
                       p =  P2 * X
         Return: P2: 3x4 left camera to image projection matrix.
    '''
    
    with open(calib_path, 'r') as f:

        for line in f:

            # remove the whitespace characs at the end
            line = line.rstrip()

            # convert string into a list of strings
            info = line.split()

            if (info[0] == 'P2:'):
                
                # convert str to float reading row-aligned data
                num_val = [float(x) for x in info[1:]]

                cam_to_img_Tr = np.asarray(num_val, np.float32)

                # transformation matrix  in homogenous coordinates
                cam_to_img_Tr = cam_to_img_Tr.reshape((3, 4))
    
    return cam_to_img_Tr

def imu_to_cam_transform(calib_path):
    '''
           To transform a point X from GPS/IMU coordinates to the 3D camera coordinates:
                       Y =  (R|T)_velo_to_cam * (R|T)_imu_to_velo * X,
           where
               - (R|T)_velo_to_cam (4x4): velodyne coordinates -> cam 0 coordinates
               - (R|T)_imu_to_velo (4x4): imu coordinates -> velodyne coordinates
           Inputs: calib_dir is the path to the calibration file.
           Output: 4x4 transformation matrix from GPS/IMU coordinates to 3D camera
                   coordinates.
       '''

    row4 = np.asarray([0, 0, 0, 1], dtype=np.float32)

    with open(calib_path, 'r') as f:

        for line in f:

            # remove the whitespace characters at the end
            line = line.rstrip()

            # convert string into a list of strings
            info = line.split()

            if (info[0] == 'Tr_velo_cam'):

                # convert str to float reading row-aligned data
                num_val = [float(x) for x in info[1:]]

                Tr_velo_cam = np.asarray(num_val, np.float32)

                # transformation matrix  in homogenous coordinates
                Tr_velo_cam = np.vstack((Tr_velo_cam.reshape((3, 4)), row4))

            elif (info[0] == 'Tr_imu_velo'):

                # convert str to float reading row-aligned data
                num_vals = [float(x) for x in info[1:]]

                Tr_imu_velo = np.asarray(num_vals, np.float32)

                # transformation matrix  in homogeneous coordinates
                Tr_imu_velo = np.vstack((Tr_imu_velo.reshape((3, 4)), row4))

                # transform a point X from GPS/IMU coordinates to the camera coordinates
                
    Tr_imu_cam = np.dot(Tr_imu_velo, Tr_velo_cam)

    return Tr_imu_cam

def mercator_projection(lat,lon,att,lat0):
    '''
        Project the GPS/IMU measurements to a planar map.
    '''
    r = 6378137

    s = np.cos(lat0*np.pi/180.0)

    # east position along the equator
    pos_x = s*r*np.pi*lon/180

    # north position wrt central meridian
    pos_y = s*r*np.log(np.tan(np.pi*(0.25 + lat/360)))

    # altitude is same as z
    pos_z = att 

    return np.asarray([[pos_x, pos_y, pos_z]], dtype = np.float64)

def get_rot_mat(rx,ry,rz):
    '''
        Construct the 3D rotation matrix from oxts to the world map.
    '''

    # basic rotation matrices
    Rx = np.asarray([[1, 0, 0],
                     [0, np.cos(rx), -np.sin(rx)],
                     [0, np.sin(rx), np.cos(rx)]], dtype = np.float64)

    Ry = np.asarray([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]], dtype = np.float64)

    Rz = np.asarray([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]], dtype = np.float64)

    R  =  functools.reduce(np.dot, [Rz,Ry,Rx])

    return R

def concatenate(R,t):

    # rotation and translation matrix [R|t]
    RT = np.hstack((R,t.T))

    row4 = np.asarray([0,0,0,1], dtype = np.float32)

    # homogenous transformation matrix
    RT_h = np.vstack((RT,row4))

    return RT_h

def imu_to_world_transform(oxts_path = None,video_no = 0):
    '''
        Compute the transformation matrices which takes a 3D point in the
        ith gps/imu coordinates and projects it to world coordinates. 
         Inputs: path_oxts is the path to the location of oxts folders.
                 video no is the integer used to name the directory of
                 the oxts file.
         Outputs: projs is the list of 4x4 transformation matrices.
    '''
    if (oxts_path == None):
        oxts_path = kitti_oxts_dir  + '/{0:0>4}'.format(str(video_no)) + '.txt'

    # projection matrices to the world coordinates in the 1st frame
    projs = []

    # the latitude of the first frame's coordinates
    lat_0 = None

    try:

        # read the imu gps data for each frame for the given video
        with open(oxts_path,'r') as f:

            for k,line in enumerate(f):

                l = line.rsplit()

                gps_imu = [float(x) for x in l[0:6]]

                lat, lon, att = gps_imu[0:3]

                if (k==0 or lat_0 is None):
                    lat_0 = lat

                t = mercator_projection(lat,lon,att,lat_0)

                # rotations:
                rx = gps_imu[3]  # roll around the x-axis
                ry = gps_imu[4]  # pitch around the y-axis
                rz = gps_imu[5]  # heading around the z-axis

                R = get_rot_mat(rx,ry,rz)

                # ith oxts coordinates -> the world coordinates
                Rt_h = concatenate(R,t)

                projs.append(Rt_h)

    except IOError as e:
        print("Could not read file:{0.filename}".format(e))
        sys.exit()

    return projs

def oxts_0_frame_trans(path_oxts = None,video_no = 0):
    '''
        Compute the transformation matrices which takes a 3D point in
        the i'th frame and projects it into the oxts coordinates of the
        (i-1)st frame. In addition, compute the change in the yaw angle 
        between those two frames.
         Inputs: path_oxts is the path to the location of oxts folders.
                 video no is the integer used to name the directory of
                 the oxts file.
         Outputs: projs is the list of 4x4 transformation matrices.
                  delta_yaw is the list of changes in yaw angle, i.e.,
                  delta_yaw_i = yaw_i - yaw_(i-1).
    '''

    if (path_oxts == None):
        oxts_path = kitti_oxts_dir  + '/{0:0>4}'.format(str(video_no)) + '.txt'

    # projection matrices to the oxts coordinates in the 0th frame
    projs = []

    # change in yaw angle of the oxts between ith and 0th frames
    # delta_yaw = []

    # the latitude of the first frame's coordinates
    lat_0 = None

    # transform inertial coordinates to the 0th oxts coordinates
    Rt_0 = None

    try:

        # read the imu gps data for each frame for the given video
        with open(path_oxts,'r') as f:

            for k,line in enumerate(f):

                l = line.rsplit()

                gps_imu = [float(x) for x in l[0:6]]

                lat, lon, att = gps_imu[0:3]

                if (k==0 or lat_0 is None):
                    lat_0 = lat

                t = mercator_projection(lat,lon,att,lat_0)

                # rotations:
                rx = gps_imu[3]  # roll around the x-axis
                ry = gps_imu[4]  # pitch around the y-axis
                rz = gps_imu[5]  # heading around the z-axis

                R = get_rot_mat(rx,ry,rz)

                # ith oxts coordinates -> the world coordinates
                Rt_h = concatenate(R,t)

                # normalization matrix to start start at (0,0,0)
                if (k == 0 or Rt_0 is None):
                    Rt_0 = Rt_h
                    # delta_yaw[k] = 0

                # if k>0:
                #     delta_yaw[k] -= delta_yaw[0]

                # the world coordinates -> 0th oxts coordinates
                oxts_proj_i = np.linalg.solve(Rt_0,Rt_h)

                projs.append(oxts_proj_i)

    except IOError as e:
        print("Could not read file:{0.filename}".format(e))
        sys.exit()

    return projs
