import numpy as np

def iou(bb0_cntr, bb0_dims, bb1_cntr, bb1_dims):
    """
    Computes IoU between test bboxes and the gt_bbox in the form of [x1,y1,x2,y2].
    Input: for i in {0,1} bbi_cntr is 1D array of 2D bbox center.
           and bbi_dims is 1D array of 2D bbox dimensions.
    """
    
    bb0_x1y1 = bb0_cntr - bb0_dims/2
    bb0_x2y2 = bb0_cntr + bb0_dims/2
    bb0 = np.concatenate((bb0_x1y1,bb0_x2y2))  
    
    bb1_x1y1 = bb1_cntr - bb1_dims/2
    bb1_x2y2 = bb1_cntr + bb1_dims/2
    bb1 = np.concatenate((bb1_x1y1,bb1_x2y2))  
    
    xx1 = np.maximum(bb0[0], bb1[0])
    yy1 = np.maximum(bb0[1], bb1[1])
    xx2 = np.minimum(bb0[2], bb1[2])
    yy2 = np.minimum(bb0[3], bb1[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb0[2]-bb0[0])*(bb0[3]-bb0[1])
               + (bb1[2]-bb1[0])*(bb1[3]-bb1[1]) - wh)
    return o