import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag
from scipy import special

def bern_eq(x, f, q):
    return np.concatenate([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])

def bern_eq_solver(x, f, q):
# =============================================================================
#     x = np.array([1/qt[0, t], 1/qt[0, t]]).reshape(2,1)
#     f = ft[0, t]
#     q = qt[0, t]
# =============================================================================
    f_org = np.array([[special.polygamma(0, x[0,0]) - special.polygamma(0, x[1,0])-f], 
                  [special.polygamma(1, x[0,0]) + special.polygamma(1, x[1,0])-q]])
    Df = np.array([[special.polygamma(1, x[0,0]), -special.polygamma(1, x[1,0])],
                   [special.polygamma(2, x[0,0]), special.polygamma(2, x[1,0])]])
    
    x_new = x - np.linalg.inv(Df) @ f_org
    while (abs(x[0]/x_new[0]-1)>=1e-5 and abs(x[1]/x_new[1]-1)>=1e-5):
            x = x_new
            f_org = np.array([[special.polygamma(0, x[0,0]) - special.polygamma(0, x[1,0])-f], 
                          [special.polygamma(1, x[0,0]) + special.polygamma(1, x[1,0])-q]])
            Df = np.array([[special.polygamma(1, x[0,0]), -special.polygamma(1, x[1,0])],
                           [special.polygamma(2, x[0,0]), special.polygamma(2, x[1,0])]])
            x_new = x - np.linalg.inv(Df) @ f_org
    return x_new[0, 0], x_new[1, 0]

def pois_eq(x, q):
    return special.polygamma(1, x)-q

def pois_eq_solver(x, q):
# =============================================================================
#     x = 1/qt[0, t]
#     q = qt[0, t]
# =============================================================================
    f_org = special.polygamma(1, x)-q
    Df = special.polygamma(2, x)
    
    x_new = x - f_org / Df
    while (abs(x/x_new-1)>=1e-5):
            x = x_new
            f_org = special.polygamma(1, x)-q
            Df = special.polygamma(2, x)
            x_new = x - f_org / Df
    return x_new

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
        Ct_star = (Ct_star + Ct_star.T)/2
        state = np.random.multivariate_normal(mean = mt_star, 
                                cov = Ct_star)
        rate_samps[s, TActual - 1] = np.exp(F.T @ state)
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            # When time points aren't updated
            # Keep former parameters
            if skipped[:,t+1]:
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                rate_samps[s, t] = np.exp(F.T @ state)
            else:
                Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
                mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
                Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
                Ct_star = (Ct_star + Ct_star.T)/2
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
        Ct_star = (Ct_star + Ct_star.T)/2
        state = np.random.multivariate_normal(mean = mt_star, 
                                cov = Ct_star)
        samps[s, TActual - 1] = special.expit(F.T @ state)
        # Trajectory
        for t in np.arange(TActual-2, -1, -1):
            # When time points aren't updated
            # Keep former parameters
            if skipped[:,t+1]:
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                samps[s, t] = np.exp(F.T @ state)
            else:
                Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
                mt_star = mt[:, t] + Bt @ (state - at[:,  t + 1])
                Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
                Ct_star = (Ct_star + Ct_star.T)/2
                state = np.random.multivariate_normal(mean = mt_star, cov = Ct_star)
                samps[s, t] = special.expit(F.T @ state)
                t = t - 1
    return(samps)

def FF_Bernoulli_BK(F, G, delta, flow, n, T0, TActual, eps):
    # F = F_bern; G = G_bern; delta = delta_bern; 
    # flow = flow_count
    # Initiate: Bernoulli
    d1, d2 = G.shape
    mt  = np.zeros((d2,TActual))
    Ct  = np.zeros((d2,d2,TActual))
    at  = np.zeros((d1,TActual))
    Rt  = np.zeros((d1,d1,TActual))
    rt  = np.zeros((1,TActual))
    st  = np.zeros((1,TActual))
    ft  = np.zeros((1,TActual+1))
    qt  = np.zeros((1,TActual+1))
    sft = np.zeros((1,TActual))
    sqt = np.zeros((1,TActual))
    skipped = np.zeros((1,TActual))
    
    # Prior: Bernoulli
    # m0 = np.array([[0],
    #                [0]])
    # C0 = 0.1*np.eye(2)
    
    m0 = np.zeros((d2, 1))
    C0 = 0.1*np.eye(d2)
    
    # Transform
    zt = np.array(flow[n: (n+1), :] > eps)

    # Forward filtering: Bernoulli
    for t in range(TActual):
        # t = 0
        # When t = 1. Get prior from m0, C0
        if t == 0: 
            at[:, t]   = (G @ m0)[:,0]
            Rt[:,:,t] = G @ C0 @ G.T/delta
        else:
            at[:,t]   = (G @ mt)[:,t-1]
            Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T/delta
        
        # Stop updating when same value keeps coming back (i.e. all 1)
        # To visit: is this stopping rule right? Work for all 0?
        if t > 0 and abs(rt[:,t-1] / (rt[:,t-1] + st[:,t-1]) - zt[:,t]) < 1e-2:
            at[:,t] = at[:,t-1]
            Rt[:,:,t] = Rt[:,:,t-1]
            mt[:,t] = mt[:,t-1]
            Ct[:,:,t] = Ct[:,:,t-1]
            rt[:, t] = rt[:, t-1]
            st[:, t] = st[:, t-1]
            skipped[:, t] = True
            continue
        
        ft[:, t]     = F.T @ at[:,t]
        qt[:, t]     = F.T @ Rt[:,:,t]  @ F
        
        # Numerically approximate rt, st
# =============================================================================
#         xnew = least_squares(bern_eq, [1, 1],args = (ft[:, t], qt[:, t]), bounds = ((0, 0), (1e16, 1e16)))
#         rt[:, t] = xnew.x[0]
#         st[:, t] = xnew.x[1]
# =============================================================================
        
        rt[0, t], st[0, t] = bern_eq_solver(np.array([[1/qt[0, t]], [1/qt[0, t]]]),
                       ft[0, t], qt[0, t])
        
        # Update ft, qt
        sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, T0+t]) - special.polygamma(0, st[:, t] + 1 - zt[:, T0+t])
        sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, T0+t]) + special.polygamma(1, st[:, t] + 1 - zt[:, T0+t]);
        
        # Posterior mt, Ct
        mt[:,t]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
        Ct[:,:,t] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
    return mt, Ct, at, Rt, rt, st, skipped

def FF_Poisson_BK(F, G, delta, flow, n, mN, T0, TActual, eps, RE_rho, conditional_shift):
   
    # Incorporate Random effect: change F, G, delta, mt, Ct, at, Rt
    # Add it anyway. RE_rho = 1 is the case when I don't want RE
    # Too bad matlab cannot assign default value as I do in R
    
    # F = F_pois; G = G_pois; delta = delta_pois; RE_rho = RE_rho_pois; conditional_shift = conditional_shift_pois;
    
    #F = np.r_[np.array([[1]]), F]
    #G = block_diag(0, G)
    d1, d2 = G.shape
    delta = np.diag(np.c_[np.array([RE_rho]), np.matlib.repmat(delta,1,d2-1)][0,:])
    
    # Initialize parameters
    mt  = np.zeros((d2,TActual))
    Ct  = np.zeros((d2,d2,TActual))
    at  = np.zeros((d1,TActual))
    Rt  = np.zeros((d1,d1,TActual))
    rt  = np.zeros((1,TActual))
    ct  = np.zeros((1,TActual))
    ft  = np.zeros((1,TActual+1))
    qt  = np.zeros((1,TActual+1))
    sft = np.zeros((1,TActual))
    sqt = np.zeros((1,TActual))
    
    skipped = np.zeros((1, TActual))

    # Prior: Poisson
    # m0 = np.array([1, np.log(max(flow[n,T0-1],1)), 0]).reshape(d2, 1)
    # C0 = 0.1 * np.eye(3)
    
    m0 = np.concatenate((np.array([1, np.log(max(flow[n,T0-1],1)), 0]), 
                              np.zeros((d2-3, ))
                              )).reshape(d2, 1)
    C0 = 0.1 * np.eye(d2)

    # Retrieve up data series
    xt = flow[n, :]
    # Shift count for conditional poisson
    xt[np.where(xt == eps)] = 0
    xt = xt - conditional_shift

    
    # To revisit: First time point FORCE update: if <0. set to 0;
    if xt[T0] < 0:
        xt[T0] = 0
    
    for t in range(TActual):
        # t = 0
        # No update when xt[T0+t-1] < 0 for conditional Poisson model
        if conditional_shift != 0 and xt[T0+t] < 0:
            if t == 0:
                mt[:,t] = m0[:, 0]
                Ct[:,:,t] = C0
            else:
                at[:,t] = at[:,t-1]
                Rt[:,:,t] = Rt[:,:,t-1]

                mt[:,t] = mt[:,t-1]
                Ct[:,:,t] = Ct[:,:,t-1]

                rt[:, t] = rt[:, t-1]
                ct[:, t] = ct[:, t-1]
            
            skipped[:,t] = True
        # Update
        else:
            if t == 0:
                at[:,t]   = (G @ m0)[:, 0]
                Rt[:,:,t] = np.linalg.inv(delta) @ G @ C0 @ G.T
            else:
                at[:,t]   = (G @ mt)[:,t-1]
                Rt[:,:,t] = np.linalg.inv(delta) @ G @ Ct[:,:,t-1] @ G.T
            
            ft[:, t]      = F.T @ at[:,t]
            qt[:, t]      = F.T @ Rt[:,:,t] @ F # just for clarity
            
            # Random effect
            # BK Question: Why would we need these lines? RE already in delta?
            Rt[0,0,t] = qt[:, t] * (1-delta[0,0]) / delta[0,0] # vt
            qt[:, t]  = qt[:, t] + Rt[0,0,t] # % p.31 qt + vt. to revisit param
            
            # Get numerical root of Gamma approximation
# =============================================================================
#             rnew = least_squares(pois_eq, 1, args = (qt[:, t]), bounds = ((0), (1e16)))
#             rt[:, t] = rnew.x
#             ct[:, t] = np.exp(special.polygamma(0, rt[:, t])-ft[:, t])
# =============================================================================
            
            
            rt[:, t] = pois_eq_solver(1/qt[0, t], qt[0, t])
            ct[:, t] = np.exp(special.polygamma(0, rt[:, t])-ft[:, t])
            
            
            # sft[:, t] = special.polygamma(0, rt[:, t]+xt[T0+t]) - np.log(ct[:, t]+1)
            sft[:, t] = special.polygamma(0, rt[:, t]+xt[T0+t]) - np.log(ct[:, t]+mN[t,0])
            sqt[:, t] = special.polygamma(1, rt[:, t]+xt[T0+t])
            mt[:,t] = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t])/qt[:, t]
            Ct[:,:,t] = Rt[:,:,t]-Rt[:,:,t] @ F @ (F.T) @ Rt[:,:,t] * (1-sqt[:, t]/qt[:, t])/qt[:, t]
    return mt, Ct, at, Rt, rt, ct, skipped