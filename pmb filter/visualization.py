import os
import numpy as np
import cv2
import sys

def draw_bbox(img, n, tracks, world_to_imu_proj, cam_to_img_proj, imu_to_cam_proj, save_dir, frame_no, measurements):
    
    tracks.sort(key=lambda t: t.target.r, reverse=True)
    
    for i,t in enumerate(tracks):
        
        if (i >= n):
            break
        
        xyz1_world = np.concatenate((t.target.x[0:3],[1]))
        
        # world - > imu -> cam transformations
        xyz1_cam = imu_to_cam_proj@world_to_imu_proj@xyz1_world
 
        # cam -> image transformation
        xyz_img = cam_to_img_proj@xyz1_cam
        
        u, v = np.rint(xyz_img[0:-1] / xyz_img[-1]) 
        
        cv2.putText(img, str(t.trackId), (int(u),int(v)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [25,255,125], 2, cv2.LINE_AA)
        
    for Z in measurements:
        xyz1_world = np.concatenate((Z['state'][0:3],[1]))
        
         # world - > imu -> cam transformations
        xyz1_cam = imu_to_cam_proj@world_to_imu_proj@xyz1_world
 
        # cam -> image transformation
        xyz_img = cam_to_img_proj@xyz1_cam
        
        u, v = np.rint(xyz_img[0:-1] / xyz_img[-1]) 
        
        cv2.circle(img, center=(int(u),int(v)), radius=3, thickness=1, color=[255,0,0])
        
        
    cv2.imshow('tracking output', img)
    save_path = os.path.join(save_dir, str(frame_no) + '.png')
    cv2.imwrite(save_path,img)
    if cv2.waitKey(1) == ord("q"):
        print('pressed to quit!')
        sys.exit()

    
    
    