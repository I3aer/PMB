import cv2

import numpy as np

from single_track import track

from  marginal_data_association import lbp

from tomb import mb_hypotheses

#import data_processing

from config import state_dim, track_status, cI, lambda_b, max_rbirth, \
max_num_exist_tracks, max_num_measurements, marginal_assoc_threshold

from functools import reduce

#import multiprocess as mp

from kitti_data_simulator import simulator

from functools import partial

from visualization import draw_bbox

from kitti_coordinate_transforms import imu_to_cam_transform

import cProfile


def run_tomb_tracker():
    
    existing_tracks = [] # Multi-Bernoulli
    newborn_tracks = []  # Adaptive Poisson RFS
    
    #data_process = data_processing()
    
    #pool = mp.Pool(mp.cpu_count())
    
    #mp.freeze_support()
        
    sim = simulator(video_filename='0019')
    
    cam_to_img_proj = sim.get_cam_to_image_proj()
    
    path_calib = sim.path_calib
    
    imu_to_cam_proj = imu_to_cam_transform(path_calib)
    
    # create a window to display 
    cv2.namedWindow('tracking output', cv2.WINDOW_AUTOSIZE)
    
    # weights, existence probabilities, states, dimensions and covariances of existing MB hypotheses 
    wupd = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1),dtype=np.float32)
    rupd = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1),dtype=np.float32)
    xupd = np.zeros(shape=(max_num_exist_tracks, max_num_measurements,state_dim),dtype=np.float32)
    dupd = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1,2),dtype=np.float32)
    Pupd = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1,state_dim,state_dim),dtype=np.float32)
    
    # contributions computed for each measurements to weights, existence probabilities,
    # states, dimensions and  covariances using each Gaussian components of Poisson RFS.
    wnew = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1),dtype=np.float32)
    rnew = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1),dtype=np.float32)
    xnew = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1,state_dim),dtype=np.float32)
    dnew = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1,2),dtype=np.float32)
    Pnew = np.zeros(shape=(max_num_exist_tracks, max_num_measurements+1,state_dim,state_dim),dtype=np.float32)

    try:
        while(True):
            
            # Receive measurements:
            
            #print('frame_no:{0:d}'.format(sim.frame_no))

            img = sim.read_image()
            
            imu_to_world_proj = sim.get_imu_to_world_proj() 
            
            world_to_imu_proj = np.linalg.inv(imu_to_world_proj)
            
            measurements = sim.get_measurements()
            
            num_targets_exist = len(existing_tracks)
            num_meas = len(measurements)
            
            # Filtering:
            for i, t in enumerate(existing_tracks):
                t.predict(world_to_imu_proj, imu_to_cam_proj, cam_to_img_proj)
                t.update(measurements, wupd[i,0:num_meas+1], rupd[i,0:num_meas+1],
                         xupd[i,0:num_meas+1,:], dupd[i,0:num_meas+1,:], Pupd[i,0:num_meas+1,:,:])
            
            num_targets_newborn = len(newborn_tracks)
            
            # combine individual hypothesis of GM components for each measurement 
            if (num_targets_newborn>0 and num_meas>0):
                
                for i, t in enumerate(newborn_tracks):
                    t.predict(world_to_imu_proj, imu_to_cam_proj, cam_to_img_proj)
                    t.update(measurements, wnew[i,0:num_meas+1], rnew[i,0:num_meas+1],
                             xnew[i,0:num_meas+1,:], dnew[i,0:num_meas+1,:], Pnew[i,0:num_meas+1,:,:])
            
                # lambda_fa + <lambda_u,g*Pd> for each measurement where
                # lambda_u is Poisson GM intensity of undetected tracks
                wnew_PGM = cI + np.sum(wnew[0:num_targets_newborn,1:num_meas+1],axis=0).reshape(-1,1)
            
                rsum = np.sum(rnew[0:num_targets_newborn,1:num_meas+1],axis=0).reshape(-1,1)
            
                # GM weights for each components 
                wgm = rnew[0:num_targets_newborn,1:num_meas+1]/(np.reshape(rsum,(1,-1)) + 10**-10)

                # <lambda_u,g*Pd> / (lambda_fa + <lambda_u,g*Pd>) 
                rnew_PGM = rsum / (wnew_PGM + 10**-10)
            
                xmu = xnew[0:num_targets_newborn,1:num_meas+1,:]
            
                # merge GM spatial densities to Gaussian one
                xnew_PGM = np.sum(xnew[0:num_targets_newborn,1:num_meas+1,:]*wgm[...,np.newaxis],axis=0)
                dnew_PGM = np.sum(dnew[0:num_targets_newborn,1:num_meas+1,:]*wgm[...,np.newaxis],axis=0)

                # innovation vectors 
                inno = xnew_PGM[np.newaxis,:,:]  - xmu
                inno = inno.reshape(-1,state_dim,1)
            
                # innovation covariance inno
                P_inno = list(map(lambda x: x.dot(x.transpose()), inno))
                P_inno = np.array(P_inno).reshape(num_targets_newborn,num_meas,state_dim,state_dim)

                # sum_j wgm[:,j]*[Pnew[:,j,:,:] + P_inno[:,j,:,:]  
                Pnew_PGM = np.sum(wgm[...,np.newaxis,np.newaxis]*
                                  (Pnew[0:num_targets_newborn,1:num_meas+1,:,:] + P_inno),axis=0)

            else: # no measurement means no birth/reborn 
                for i, t in enumerate(newborn_tracks):
                    t.predict(world_to_imu_proj, imu_to_cam_proj, cam_to_img_proj)
                
                wnew_PGM = np.zeros(shape=(num_meas,1),dtype=np.float32)
                rnew_PGM = np.zeros(shape=(num_meas,1),dtype=np.float32)
                xnew_PGM = np.zeros(shape=(num_meas,state_dim),dtype=np.float32)
                dnew_PGM = np.zeros(shape=(num_meas,2),dtype=np.float32)
                Pnew_PGM = np.zeros(shape=(num_meas,state_dim,state_dim),dtype=np.float32)
                
            # LBP to estimate marginal association probabilities: 
            pupd, pnew = lbp(wupd[0:num_targets_exist,0:num_meas+1],wnew_PGM)
            
            # TOMB algorithm: form new multi-Bernoulli components using:
            x, d, r, P, global_hypo_index =  mb_hypotheses(pupd,
                                                           rupd[0:num_targets_exist,0:num_meas+1],
                                                           xupd[0:num_targets_exist,0:num_meas+1,:],
                                                           dupd[0:num_targets_exist,0:num_meas+1,:],
                                                           Pupd[0:num_targets_exist,0:num_meas+1,:,:],
                                                           pnew,
                                                           rnew_PGM, xnew_PGM, dnew_PGM ,Pnew_PGM)
            
            # estimate the number of tomb tracks
            if (r.shape[0]>0):
                # estimated number of targets
                n = int(sum(list(map(lambda x: np.rint(x),r)))[0])
            else:
                n = 0
            
            # Generate adaptive birth RFS:
            
            # compute the probability of a measurement originated from a newborn target
            rbirth = np.zeros((num_meas,1))
            if (num_targets_exist > 0):
                rbirth = np.sum(pupd[:,1:], axis=0).reshape(-1,1)
            if (num_targets_newborn > 0):
                rbirth += pnew
            rbirth = np.minimum((1 - rbirth), max_rbirth)
    
            birth_tracks = [track(Z,r[0]) for r,Z in zip(rbirth,measurements) if r[0] > 2*marginal_assoc_threshold]

            # Track management:
            
             # determine major and minor track hypotheses
            if (global_hypo_index.size>0):
                # label minor hypotheses in xupd and xnew
                minor_global_hypo_idx = np.logical_not(global_hypo_index)

                idx_minor_exist = minor_global_hypo_idx[0:num_targets_exist]

                idx_minor_newborn = minor_global_hypo_idx[num_targets_exist:]
            
            # tracks from TOMB
            tracks_tomb = []
            fn_tomb_append = tracks_tomb.append

            # index used for x,r and P from the TOMB filter
            idx_major_hypo = 0
             
            for i in range(num_targets_exist):
 
                t = existing_tracks[i]
              
                # if a minor track neglect/terminate it it
                if (idx_minor_exist[i]):
                    existing_tracks[i] = None
                    continue
                
                # replace old hypothesis with reformed single one
                t.copy(x[idx_major_hypo,:], P[idx_major_hypo,:,:],
                       r[idx_major_hypo,:], 1, track_status.EXISTING)
                
                idx_major_hypo += 1
                
                fn_tomb_append(t)
    
            existing_tracks = [t for t in existing_tracks if t != None]
            
            fn_exist_append = existing_tracks.append
            # check if any newborn track initiated from PPP
            if np.any(global_hypo_index[num_targets_exist:]):
    
                for i in range(num_meas):
                    
                    # if a minor track hypothesized from PPP neglect/terminate it
                    if (idx_minor_newborn[i]):
                        continue
 
                    t = track(measurements[i])
                    t.copy(x[idx_major_hypo,:], P[idx_major_hypo,:,:], 
                           r[idx_major_hypo,:], 1, track_status.EXISTING)    
                    
                    idx_major_hypo += 1
                    
                    fn_exist_append(t)
                    fn_tomb_append(t)
            
            # PPP components 
            newborn_tracks = []
            fn_newborn_append = newborn_tracks.append
            # separate births to existing and undetected tracks
            for t in birth_tracks:
                if (t.status == track_status.EXISTING):
                    fn_exist_append(t)
                    fn_tomb_append(t)
                    n += int(round(t.target.r))
                elif(t.status == track_status.NEWBORN):
                    fn_newborn_append(t)
                    
            # Visualization:
            
            # draw bounding boxes of the most promising n tracks
            draw_bbox(img, n, tracks_tomb, world_to_imu_proj, cam_to_img_proj, 
                      imu_to_cam_proj, sim.save_dir, sim.frame_no-1, measurements)
        
            xupd[0:num_targets_exist,0:num_meas+1,:].fill(0)
            dupd[0:num_targets_exist,0:num_meas+1,:].fill(0)
            wupd[0:num_targets_exist,0:num_meas+1].fill(0)
            rupd[0:num_targets_exist,0:num_meas+1].fill(0)
            Pupd[0:num_targets_exist,0:num_meas+1,:,:].fill(0)
            
            xnew[0:num_targets_newborn,0:num_meas+1,:].fill(0)
            dnew[0:num_targets_newborn,0:num_meas+1,:].fill(0)
            wnew[0:num_targets_newborn,0:num_meas+1].fill(0)
            rnew[0:num_targets_newborn,0:num_meas+1].fill(0)
            Pnew[0:num_targets_newborn,0:num_meas+1,:,:].fill(0)

    finally:
        #pool.close()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    
    #cProfile.run('run_tomb_tracker()')
    run_tomb_tracker()