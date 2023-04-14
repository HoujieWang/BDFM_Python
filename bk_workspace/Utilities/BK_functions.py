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
        mt_star = mt[: ,TActual - 1]
        Ct_star = Ct[:,: ,TActual - 1]
        state = np.random.multivariate_normal(mean = mt_star, 
                                cov = Ct_star)
        rate_samps[s, TActual - 1] = np.exp(F.T @ state)
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            #Skip the time points not updated
            if skipped[:,t+1]:
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                rate_samps[s, t] = np.exp(F.T @ state)
            else:
                Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
                mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
                Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                rate_samps[s, t] = np.exp(F.T @ state)
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
        mt_star = mt[: ,TActual - 1]
        Ct_star = Ct[:,: ,TActual - 1]
        state = np.random.multivariate_normal(mean = mt_star, 
                                cov = Ct_star)
        samps[s, TActual - 1] = special.expit(F.T @ state)
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            #Skip the time points not updated
            if skipped[:,t+1]:
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                samps[s, t] = np.exp(F.T @ state)
            else:
                Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
                mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
                Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                samps[s, t] = special.expit(F.T @ state)
    return(samps)

