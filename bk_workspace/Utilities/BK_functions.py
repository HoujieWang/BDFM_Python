import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
from scipy import special

def RA_Poisson_BK(TActual, F, G, mt, Ct, at, Rt, skipped, nSample):
    # F = F_pois; G = G_pois; mt = mt_pois; Ct = Ct_pois; 
    # at = at_pois; Rt = Rt_pois; skipped = skipped_pois;
   
    # Change F, G and delta for random effect
    #F = np.r_[np.array([[1]]), F]
    #G = block_diag(0, G)
    d1, d2 = G.shape
    
    rate_samps = np.zeros((nSample, TActual))

    for s in np.arange(0, nSample):
        # Starting Seed
        state = multivariate_normal.rvs(mean = mt[: ,TActual - 1], 
                                cov = Ct[:,: ,TActual - 1])
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            #Skip the time points not updated
            #if skipped[:,t+1]:
                #sat[:,t] = sat[:,t+1]
                #sRt[:,:,t] = sRt[:,:,t+1]
                #continue
            Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
            
            mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
            Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
            #mt_star = (1 - delta) * mt[:, t] + delta * np.linalg.inv(G) @ state
            #Ct_star = (1 - delta) * Ct[:, :, t]
            state = multivariate_normal.rvs(mean = mt_star, cov = Ct[:, :, t]) #Ct_star not PD
            rate = np.exp(F.T @ state)
            rate_samps[s, t] = rate
    return(rate_samps)

def RA_Bernoulli_BK(TActual, F, G, mt, Ct, at, Rt, skipped, nSample):
    # F = F_pois; G = G_pois; mt = mt_pois; Ct = Ct_pois; 
    # at = at_pois; Rt = Rt_pois; skipped = skipped_pois;
   
    # Change F, G and delta for random effect
    #F = np.r_[np.array([[1]]), F]
    #G = block_diag(0, G)
    d1, d2 = G.shape
    
    samps = np.zeros((nSample, TActual))

    for s in np.arange(0, nSample):

        # Starting Seed
        state = multivariate_normal.rvs(mean = mt[: ,TActual - 1], 
                                cov = Ct[:,: ,TActual - 1])
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            #Skip the time points not updated
            #if skipped[:,t+1]:
                #sat[:,t] = sat[:,t+1]
                #sRt[:,:,t] = sRt[:,:,t+1]
                #continue
            Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
            mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
            Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
            state = multivariate_normal.rvs(mean = mt_star, cov = Ct_star)
            prob = special.expit(F.T @ state)
            samps[s, t] = prob
        return(samps)

