from filterpy.kalman import MerweScaledSigmaPoints
from enum import Enum
from scipy.stats import chi2

kitti_label_dir = 'C:/Users/Owner/Desktop/Kitti/Kitti_tracking/training/label_02'
kitti_image_dir = 'C:/Users/Owner/Desktop/Kitti\Kitti_tracking/training/image_02'

kitti_oxts_dir = 'C:/Users/Owner/Desktop/Kitti/data_tracking_oxts/training/oxts'
kitti_calib_dir = 'C:/Users/Owner/Desktop/Kitti/Kitti_tracking/data_tracking_calib/training/calib'

outputs_dir = 'C:/Users/Owner/Eclipse_workspace/CamRadarMoT/MOT/pmb filter/outputs'


class track_status(Enum):
    '''
        Track status: Newborn=0, Existing=1 or Undetected=2
    '''
    NEWBORN = 0
    EXISTING = 1
    UNDETECTED=2
    
# dimension of state vector
state_dim = 6
# dimension of measurement vector
meas_dim = 4

# time step constant/kf update rate
step_dt = 0.1

# parameters of sigma point selection
ukf_alpha=0.01
ukf_beta=4.
ukf_kappa=1e5

# sigma points 
points = MerweScaledSigmaPoints(state_dim, alpha=ukf_alpha, beta=ukf_beta, kappa=ukf_kappa)

# initial estimates of unobserved states (vx, vy, vz)
unobserved_states_est = [0, 0, 0]

# variance of white noise acceleration for vehicles
vehicle_white_noise_acc_var_x = 1500 #(m/sec^2)^2
vehicle_white_noise_acc_var_y = 1000 #(m/sec^2)^2
vehicle_white_noise_acc_var_z = 500 #(m/sec^2)^2

# variance of white noise acceleration for vehicles
person_white_noise_acc_var_x = 500 #(m/sec^2)^2
person_white_noise_acc_var_y = 500 #(m/sec^2)^2
person_white_noise_acc_var_z = 100 #(m/sec^2)^2

# measurement uncertainties of detector and radar
detect_var_x = 16e-4 #m^2
detect_var_y = 16e-4 #m^2
detect_var_z = 25e-4 #m^2
radar_var_r = 16e-4   #m^2

# coefficient to scale camera cov matrix to initialize P0
cam_cov_coeff_xyz = 10
cam_cov_coeff_vel = 100

# track survival probability
pS = 0.9
# object detection probability
pD = 0.9

# chi2 gating distance 
gate_distance_vehicle = chi2.ppf(df=meas_dim,q=0.99)
gate_distance_pedestrian = gate_distance_vehicle/7.

# 2D bbox threshold
iou_threshold = 1e-1

# minimum threshold for association
marginal_assoc_threshold = 1e-1

# existence threshold for pruning estimated Bernoulli hypotheses
prune_target_hypo = 5e-2

# convergence threshold for loopy belief propagation (lbp)
conv_threshold = 1e-2

# maximum number of iterations for lbp
max_itr_lbp = 100

# maximum number of tracks existing
max_num_exist_tracks = 40

# maximum number of measurements
max_num_measurements = 50

# Cuboid ROI centered the camera coordinates
roi_length = 70. #m
roi_width = 40.  #m
roi_height = 3.  #m

# clutter intensity
lambda_fa = 1.
pz = 1./(roi_height*roi_width*roi_height)
cI = lambda_fa*pz

# target birth weight
max_rbirth = 1.
# expected number of target birhts
lambda_b = 1.

# the threshold probability to declare existing track
pexist = 0.6