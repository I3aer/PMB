from filterpy.kalman import UKF as ukf
from filterpy.kalman import kalman_filter as kf
from filterpy.kalman import unscented_transform as UT
from filterpy.stats import logpdf

import numpy as np

from math import exp

from config import *
from utils import iou as iou_func

from state_space_models import CVModel, measModel

from functools import partial, reduce
from itertools import starmap

class target_hypothesis():
    '''
        single target hypothesis.
    '''
    
    def __init__(self, state, bbox_dims, cov_matrix, exist_prob, weight):

        self.x = state.copy()
        self.P = cov_matrix.copy()
        self.r = exist_prob
        self.w = weight
        self.d = bbox_dims.copy()

class track(object):
    '''
        Track class is used to track a specific target in time.
        Target state is given by [x, y,z, vx, vy, vz, w, h] 
        where (x,y,z) is the 3D position in camera coordinates, 
        (vx, vy, vz) are the corresponding velocities, (w,h) is
        is the 2D bbox dimensions. 
        
        Measurement vector is formed after measurement fusion and 
        consists of camera detection [x, y, z, w, h]  and radar 
        range detection [r] wrt camera, i.e., [x, y, z, w, h, r].
    '''
    # class/static variable used to uniquely label each track
    id_counter = 0

    def __init__(self, Z, rbirth=1.):
        '''
            Initialize a track for an unassociated measurement.
            Inputs: z is the dictinary consisting of
                    Z['state'] is the [x, y, z, w, h, r] measurement
                    vector from measurement fusion. 
                    Z['cls_prob'] is the classification probability 
                    from video detector.
                    Z['conf_prob] is the confidence probability of 
                    radar detection.
                    Z['object_class'] is the predicted class of this 
                    object from video object detector.
                    rbirth is the target birth probability from the 
                    given Z.
        '''
        
        # class/type of the track
        self.obj_class = Z['object_class']
        
        if (self.obj_class == 'Pedestrian'):
            motion_noise = np.tile([person_white_noise_acc_var_x,
                                    person_white_noise_acc_var_y,
                                    person_white_noise_acc_var_z,],2)
        else:
            motion_noise = np.tile([vehicle_white_noise_acc_var_x,
                                    vehicle_white_noise_acc_var_y,
                                    vehicle_white_noise_acc_var_z,],2)
        
        self.motion_model = CVModel(motion_noise, step_dt)
        
        self.measurement_model = measModel(detect_var_x,
                                           detect_var_y,
                                           detect_var_z,
                                           radar_var_r)
        
        self.cam_R, self.radar_R = self.measurement_model.get_R()
    
        # measurement noise cov matrix
        self.R = np.concatenate((self.cam_R, self.radar_R))
        
        # measurement vector
        z = Z['state']
        
        # initial state vector using video detection
        x0 = np.array([*z[0:3], *unobserved_states_est],dtype=np.float32)
        
        # set the initial system covariance matrix P0
        P0 = np.zeros(shape=(state_dim, state_dim), dtype=np.float32)
        P0[0:3, 0:3] = cam_cov_coeff_xyz * self.cam_R[0:3, 0:3]
        P0[3:, 3:] = cam_cov_coeff_vel * self.cam_R[0:3, 0:3]
       
        # set a unique track Id 
        self.trackId = track.id_counter
        # increase the track Id by 1
        track.id_counter += 1
        
        #  update initial state by associated radar detection
        if (z.shape[0] == meas_dim):
            self.filt = ukf.UnscentedKalmanFilter(dt=step_dt, dim_x=state_dim, dim_z=1,
                                                  points=points, fx=None, hx=self.measurement_model.g)
            
            # a priori states and its covariance
            self.filt.x = x0
            self.filt.P = P0
                        
            # calculate sigma points for given mean and covariance
            self.filt.sigmas_f = self.filt.points_fn.sigma_points(x0, P0)
            
            # range update via UKF
            self.filt.update(z[-1].reshape(1))
            
            # updated states and its covariance
            x0 = self.filt.x
            P0 = self.filt.P
            
            logllk = self.filt.log_likelihood
            
            # compute <lambda(x),g*pD(x)> where lambda(x)=cls_prob*fx and pD(x)=pD*conf_prob  
            pD_pConf_llk = Z['cls_prob']*Z['conf_prob']*pD*exp(logllk)
            
            # update of the detected Poisson target
            weight = cI + pD_pConf_llk 
            exist_prob = pD_pConf_llk / weight
            
            if ( rbirth > pexist):
                self.status = track_status.EXISTING 
            else:
                self.status = track_status.NEWBORN 
            
        else: #no radar detection
            
            weight = 1.
            # cls_prob = E[lambda(x)]
            exist_prob = Z['cls_prob']
            
            self.status = track_status.NEWBORN
            
        self.filt = ukf.UnscentedKalmanFilter(dt=step_dt, dim_x=state_dim, dim_z=meas_dim,
                                              points=points, fx=None, hx=self.measurement_model.h)
        
        # set measurement covariance matrix
        self.filt.R = self.R
            
        self.target = target_hypothesis(x0, Z['bbox_dims'], P0, exist_prob, weight)
        
        if (self.obj_class == 'Pedestrian'):
            self.gating_threshold = gate_distance_pedestrian
        else:
            self.gating_threshold = gate_distance_vehicle
             

    def predict(self, world_to_imu_proj, imu_to_cam_proj, cam_to_img_proj):
        '''
            Predict existing/newborn track according to system model.
        '''

        # predict the state according to CV motion model
        self.target.x, self.target.P = kf.predict(self.target.x, 
                                                  self.target.P, 
                                                  self.motion_model.F, 
                                                  self.motion_model.Q)
        
        self.target.r *= pS
        
        # compute the predicted bbox center in image coordinates
        xyz1_world = np.concatenate((self.target.x[0:3],[1]))
        
        # world - > imu -> cam transformations
        xyz1_cam = imu_to_cam_proj@world_to_imu_proj@xyz1_world
 
        # cam -> image transformation
        xyz_img = cam_to_img_proj@xyz1_cam
        
        self.bbox_cntr = np.rint(xyz_img[0:-1] / xyz_img[-1])
              
    def __job_update(self,Z):
        
            # compute intersection of union between bboxes
            iou = iou_func(self.bbox_cntr, self.target.d, Z['bbox_cntr'], Z['bbox_dims'])
    
            if (Z['object_class'] != self.obj_class):            
                return 0, 0, np.zeros_like(self.target.x), np.zeros_like(self.target.P), np.zeros_like(self.target.d)
                
            self.filt.x = self.target.x
            self.filt.P = self.target.P
                
            # Unscented Kalman update according to measurement model h
            valid_status = self.filt.update(Z['state'],Th=self.gating_threshold)
            
            # check validation status
            if not(valid_status):
                return 0, 0, np.zeros_like(self.target.x), np.zeros_like(self.target.P), np.zeros_like(self.target.d)
            
            # probability of detecting the given target
            pD_x = pD*Z['cls_prob']*Z['conf_prob']*max(iou,1e-2)
            
            llk = exp(self.filt.log_likelihood)
            
            if (self.status == track_status.EXISTING):
                # update existing probability for Bernoulli target
                exist_prob = 1
                # update weight of the hypothesis
                weight= self.target.w * (self.target.r*pD_x*llk)
            else: # for undetected/newborn Gaussian component of GM Poisson 
                # unnormalized contribution to existing probability of a measurement-oriented track 
                exist_prob = (self.target.r*pD_x*llk)
                # weight contribution of the GM component for a measurement-oriented track 
                weight = (self.target.r*pD_x*llk)
                
            rupd = exist_prob
            wupd = weight
            xupd = self.filt.x
            pupd = self.filt.P
            dupd = Z['bbox_dims']
            
            return exist_prob, weight, self.filt.x, self.filt.P, Z['bbox_dims']
                 
    def update(self, Z, wupd, rupd, xupd, dupd, pupd):
        '''
            Compute missed detection and updated hypothesis. 
        '''   
        
        num_Z = len(Z)
            
        # Missed detection hypothesis
        _x = self.target.x
        _P = self.target.P
        _r = self.target.r
        _w = self.target.w
        _d = self.target.d

        # missed detection hypotheses:
            
        if (self.status == track_status.EXISTING):
            # for a symmetric positive-definite matrix 
            _P = 0.5*(_P + _P.transpose())  
            _w = _w*(1 - _r*pD) 
            _r = _r * (1 - pD) / ( 1 - _r*pD)

            # missed detection hypothesis
            wupd[0] = 1 - pD
            rupd[0] = _r
            xupd[0,:] = _x
            pupd[0,:,:] = _P
            dupd[0,:] = _d
                
        # If we don't get any measurements we are done with update step here!
        if (num_Z == 0):
            self.status = track_status.UNDETECTED
            return wupd, rupd, xupd, pupd
        
        # calculate sigma points for given mean and covariance
        self.filt.sigmas_f = self.filt.points_fn.sigma_points(self.target.x, self.target.P)
 
        # generate hypotheses for detected targets
        for i in range(num_Z):
            rupd[i+1], wupd[i+1], xupd[i+1,:], pupd[i+1,:,:], dupd[i+1,:] = self.__job_update(Z[i])
                

        return wupd, rupd, xupd, dupd, pupd
        
    def copy(self,x,P,r,w,status):
        '''
            copy the corresponding global state, covariance, and
            existence to target state, covariance and existence.

        '''  
        self.target.x = x.copy()
        self.target.P = P.copy()
        self.target.r = r.copy()
        self.target.w = r.copy()
        self.status = status
