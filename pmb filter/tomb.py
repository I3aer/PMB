import numpy as np

from config import prune_target_hypo, state_dim, marginal_assoc_threshold

def mb_hypotheses(pupd, rupd, xupd, dupd, Pupd, pnew, rnew, xnew, dnew, Pnew):
    '''
        Extract single target global hypotheses tracks from MB tracks.
        Inputs:
            pupd is a 2D nx(m+1) array of marginal association probabilities 
            between m measurements and n tracks computed by lbp.
            rupd is a 2D nx(m+1) existence probabilities of association
            hypotheses for n tracks.
            xupd is a 3D array nx(m+1)xstate_dim of state estimates for each 
            association hypotheses for all tracks.
            Pupd is a 3D array  nx(m+1)xstate_dimxstate_dim of covariance 
            estimates for each association hypotheses for all tracks.
            
            pnew is a 1D (mx1) array of marginal association probabilities 
            of newborn tracks found by lbp.
            rnew is a 1D (mx1) array of existence probabilities of newborn
            tracks.
            xupd is a 3D array mxstate_dim of new state estimates for each 
            newborn Bernoulli track.
            Pnew is a 3D array  mxstate_dimxstate_dim of covariance estimates
            of newborn Bernoulli tracks.
            
            pool is the multiprocessing pool object to distribute tasks across
            processes.
            
        return: states, existences, covariance matrices and index of promising 
                MB hypotheses x, r, P after pruning insignificant hypotheses
    '''
    
    n = pupd.shape[0]
    m = pnew.shape[0]

    # Form continuing tracks:
    if (n>0):
    
        # remove associations of minor MB hypotheses
        pupd[pupd<marginal_assoc_threshold] = 0
        # normalize marginal associations probabilities
        pupd /= (np.sum(pupd,axis=1).reshape(-1,1) + 10**-10)
    
        # compute existence probabilities
        Pr = pupd * rupd
        r_exist = np.sum(Pr,axis=1)[:,np.newaxis]
        
        Pr = Pr / (r_exist + 10**-10)
        
        # compute sum_j p(i,j)*r(i,j)*x(i,j) for each track i
        x_exist = np.sum(xupd * Pr[...,np.newaxis], axis=1)
        # compute sum_j p(i,j)*r(i,j)*d(i,j) for each track i
        d_exist = np.sum(dupd*Pr[...,np.newaxis], axis=1)
        
        # innovation vectors for nx(m+1) predictions
        inno = x_exist[:,np.newaxis,:]  - xupd
        
        inno = inno.reshape(-1,state_dim,1)
        
        P_inno = list(map(lambda x: x.dot(x.transpose()), inno))
        
        P_inno = np.array(P_inno).reshape(n,m+1,state_dim,state_dim)
        
        P_exist = np.sum(Pr[...,np.newaxis,np.newaxis]*(Pupd + P_inno),axis=1)
    else:
        x_exist = np.zeros_like(xupd)
        d_exist = np.zeros_like(dupd)
        P_exist = np.zeros_like(Pupd)
        r_exist = np.zeros(shape=(n,1))

    # Form newborn tracks:
    if (m>0):
        # remove associations of minor MB hypotheses
        pnew[pnew<marginal_assoc_threshold] = 0
        r_new = pnew * rnew
    else:
        r_new = np.zeros_like(rnew)
    
    if (n>0 and m>0):
        # states of MB hypotheses with shape of (n+1+m)xstate_dim
        x = np.concatenate((x_exist,xnew),axis=0)
        # 2D bbox dimensions of MB hypotheses with shape of (n+1+m)X2
        d = np.concatenate((d_exist,dnew),axis=0)
        # existence probabilities of (n+1+m) tracks
        r = np.concatenate((r_exist,r_new),axis=0)
        # cov matrices of MB hypotheses with shape of (n+1+m)xstate_dimxstate_dim
        P = np.concatenate((P_exist,Pnew),axis=0)
    elif (n>0):
        x = x_exist
        d = d_exist
        r = r_exist
        P = P_exist
    elif (m>0):
        x = xnew
        d = dnew
        r = rnew
        P = Pnew
    else:
        x = np.zeros(shape=(0,x_exist.shape[1]))
        d = np.zeros(shape=(0,d_exist.shape[1]))
        r = np.zeros(shape=(0,1))
        P = np.zeros(shape=(0,x_exist.shape[1],x_exist.shape[1]))
        
        return x, d, r, P, np.array([])
        
    
    # Truncate tracks with low probability of existence 
    idx = np.reshape(r > prune_target_hypo,-1) 
    r = r[idx,:]
    x = x[idx,:]
    d = d[idx,:]
    P = P[idx,:,:]
    
    return x, d, r, P, idx
