import numpy as np


class CVModel:
    '''
        Nearly constant velocity motion model.
        state vector = [x, y, z, v_x, v_y, vz] 
    '''

    def __init__(self, motion_noise, dt):

        self.q = motion_noise

        self.model = 1

        # The discretized CV model with sampling period dt: x(k + 1) = Fx(k) + v(k)
        self.F = np.array([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0,  0],
                           [0, 0, 0, 0, 1,  0],
                           [0, 0, 0, 0, 0,  1]], dtype=np.float32)

        # covariance matrix of noise v(k)
        self.Q = np.array([[dt ** 3 / 3, 0, 0, dt ** 2 / 2, 0, 0],
                           [0, dt ** 3 / 3, 0, 0, dt ** 2 / 2, 0],
                           [0, 0, dt ** 3 / 3, 0, 0, dt ** 2 / 2],
                           [dt ** 2 / 2, 0, 0, dt, 0, 0],
                           [0, dt ** 2 / 2, 0, 0, dt, 0],
                           [0, 0, dt ** 2 / 2, 0, 0, dt]],dtype=np.float32) * self.q
    
        
class measModel():
    '''
        Measurement model Z = h(X) where Z=[x, y, z, r]
        where x,y,z are 3D positions from camera, and r 
        is range from sensor (camera/imu) to object.
    '''
    
    def __init__(self, var_x, var_y, var_z, var_r):
        
        self.model = 1 
        
        self.cam_R = np.array([[var_x, 0, 0, 0],
                               [0, var_y, 0, 0],
                               [0, 0, var_z, 0]], dtype=np.float32)
        
        self.radar_R = np.array([[0, 0, 0, var_r]],dtype=np.float32) 
        

    def h(self, state):
        '''
            Nonlinear measurement function h(X) = [x, y, z, w, h, r]
            where range from camera r is computed as sqrt(x^2+y^2+z^2).
        '''
        
        x,y,z = state[0:3]

        r = np.sqrt(x**2+y**2+z**2)

        return np.array([x, y, z, r], dtype=np.float32)
    
    def g(self,state):
        '''
           Nonlinear measurement function h(X) = [r] that maps target state
           X to range measurement r, i.e, r = sqrt(x^2+y^2+z^2). This measurement
           function is used to combine camera and radar measurements at time t=0.
        '''
        
        x,y,z = state[0:3]

        r = np.sqrt(x**2+y**2+z**2)

        return np.array([r], dtype=np.float32)
        
    
    def get_R(self):
        '''
            return camera and radar measurement covariances.
        '''
        
        return self.cam_R, self.radar_R