 class data_processing():
    '''
        i) receive rgbd data from realsense
        ii) detect and classify objects
        iii) calculate depths of detected objects
        iv) backproject 2D detections to 3D X,Y,Z camera coordinates
        v) generate a list of the measurement dictionaries 
           with keys=['state','cls','cls_prob','conf_prob']
               
    ''' 
    def __init__(self):
        pass
      
    def get_measurements(self):
        pass
         
 
