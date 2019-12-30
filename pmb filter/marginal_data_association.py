import numpy as np
from config import conv_threshold, max_itr_lbp

def lbp(wupd, wnew):
    '''
        This is the python implementation of the loop belief propagation
        (lbp) to estimate marginal association probabilities on graphs 
        containing loops. lpb is a message passing algorithm between 
        nodes on a factor graph. In factor graph variable nodes represent
        association variables and factor nodes represent their marginal 
        associations or dependencies like redundancy between target asso-
        ciations a_i = j and measurement association b_j = i. The lbp algo-
        rithm iterates until the maximum change between subsequent messages
        is less than a pre-set threshold. Actually, the algorithm aims to
        minimize an approximation of the convex function called Gibbs free
        energy which is one of the term of the joint entropy of the distri-
        bution (of association events).  
        Inputs: 
                wupd is nx(m+1) array of association weights where n is
                the number of tracks, m is the number of measurements.
                wupd[i,0] is the missed detection likelihood (1-pD) and
                wupd[i,j] is the PDA likelihood pD*g(z_j|mu_i).
                wnew is 1D mx1 array of newborn track weights: wnew(j)
                = lambda_fa(z_j) + lambda_u(z_j) where lambda_u is the
                birth intensity.
        Returns: pupd: (nxm+1) array of marginal association probabilities of
                 existing tracks and 
                 pnew: (mx1) array of the marginal association probabilities of
                 newborn/reborn tracks.
    '''
    
    # marginal association probabilities
    pupd = np.zeros_like(wupd)
    pnew = np.zeros_like(wnew)
    
    # number of tracks
    n = wupd.shape[0]
    # number of measurements
    m = wupd.shape[1] - 1
    
    if (n == 0 and m == 0):
        return pupd, pnew
    
    elif (m == 0):
        return np.ones_like(wupd),pnew
    
    elif (n == 0):
        return pupd, np.ones_like(wnew)

    # message sent from node b^j to a^i
    mu = np.ones((n,m), dtype=np.float32)
    # message sent from node a^i to b^i
    nu = np.zeros_like(mu)
    # old messages 
    mu_old = np.zeros_like(mu)
    
    '''
      Run LBP iteration until error between subsequent
      messages is less than the convergence threshold.
    ''' 
    cntr = 0
    while (np.max(np.abs(mu-mu_old)) > conv_threshold):
        
        if (cntr >= max_itr_lbp):
            break
  
        mu_old = mu.copy()
        
        if (n>0):
            # indivial messages sent from b^j to a^i
            PRD = wupd[:,1:]*mu
            # total message sent to a^i from all b^j=i 
            S = wupd[:,0] + np.sum(PRD,axis=1)
            # new messages sent from a^i to each b^j 
            nu = wupd[:,1:] / (S[:,np.newaxis] - PRD + 10**-10)
  
        '''
        for i in range(n):
            prd = np.multiply(wupd[i,1:],mu[i,:])
            # w0 + sum_j=0 mu[i,j) * wupd[i,j+1]
            s = wupd[i,0] + sum(prd);
            nu[i,:] = wupd[i,1:] / (s - prd)
        '''
        if (m>0):
            # add messages from newborn nodes 
            S = wnew + np.sum(nu,axis=0)[:,np.newaxis]
            # new messages sent from b^j to a^i
            mu = 1 / (S.transpose() - nu + 10**-10)
            
        '''
        for j in range(m):
            s = wnew[j] + sum(nu[:,j])
            mu[:,j] = 1/(s - nu[:,j])
        '''
            
        cntr += 1


    # Calculate marginal association probabilities:
        
    # for existing tracks:
    S = wupd[:,0] + np.sum(wupd[:,1:]*mu,axis=1)
    pupd[:,0] = wupd[:,0] / ( S + 10**-10)
    pupd[:,1:] = wupd[:,1:] * mu / (S[:,np.newaxis] + 10**-10) 
    
    '''
    for i in range(n)
      s = wupd[i,0] + np.sum(wupd[i,1:]*mu[i,:], axis=1)
      pupd[i,0] = wupd[i,0]/s
      pupd[i,1:] = wupd[i,1:]*mu[i,:]/s
    '''
    
    # for newborn tracks
    S = wnew + np.sum(nu,axis=0)[:,np.newaxis]
    pnew = wnew / (S + 10**-10)
 
    '''
    for j in range(m),
      s = wnew[j] + np.sums(nu[:,j], axis=0);
      pnew[j] = wnew[j]/s;
    '''
    
    return pupd, pnew
