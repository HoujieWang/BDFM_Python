import numpy as np
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

    
def FF_Bernoulli(F, G, delta, flow, n, T0, TActual, eps):
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
        qt[:, t]     = Rt[0,0,t] * F.T @ F
        
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

def RA_Bernoulli(TActual, F, G, mt, Ct, at, Rt, skipped, nSample):
    
    # F = F_bern; G = G_bern; mt = mt_bern; Ct = Ct_bern; at = at_bern; Rt = Rt_bern; skipped = skipped_bern;
    # Initialization
    d1, d2 = G.shape
    sat = np.zeros((d1,TActual))
    sRt = np.zeros((d1,d1,TActual))
    ssft = np.zeros((1,TActual))
    ssqt = np.zeros((1,TActual))
    ssrt = np.zeros((1,TActual))
    ssst = np.zeros((1,TActual))
   
    # Last time point
    sat[:,TActual-1] = mt[:,TActual-1]
    sRt[:,:,TActual-1] = Ct[:,:,TActual-1]
    # Retrospective Analysis
    for t in np.arange(TActual-2, -1, -1):
        # t = 361
        # Skipp the time points not updated
        if skipped[:,t+1]:
            sat[:,t] = sat[:,t+1]
            sRt[:,:,t] = sRt[:,:,t+1]
            continue
        
        
        Bt = Ct[:,:,t] @ G.T @ np.linalg.inv(Rt[:,:,t+1])
        sat[:,t] = mt[:,t] - Bt @ (at[:,t+1] - sat[:,t+1])
        sRt[:,:,t] = Ct[:,:,t] - Bt @ (Rt[:,:,t+1] - sRt[:,:,t+1]) @ Bt.T
    

    # Gamma approximation
    for t in range(TActual):
        # t = 0
        ssft[:,t]     = F.T @ sat[:,t]
        ssqt[:,t]     = F.T @ sRt[:,:,t] @ F
        # fprintf('t = %d, ssft = %.2f, ssqt = %.2f\n', t, ssft(t), ssqt(t));
        
        # Numerically approximate rt, st
# =============================================================================
#         xnew = least_squares(bern_eq, [1, 1],args = (ssft[:, t], ssqt[:, t]), bounds = ((0, 0), (1e16, 1e16)))
#         ssrt[:,t]  = xnew.x[0]
#         ssst[:,t]  = xnew.x[1]
# =============================================================================
        
        ssrt[0, t], ssst[0, t] = bern_eq_solver(np.array([[1/ssqt[0, t]], [1/ssqt[0, t]]]),
                       ssft[0, t], ssqt[0, t])
    prob_sample = special.expit(np.random.normal(loc=ssft, \
                                                 scale=np.sqrt(ssqt), \
                                                 size=(nSample, ssft.shape[1])))
        
    return sat, sRt, ssrt, ssst, prob_sample


