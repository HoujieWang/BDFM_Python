import numpy as np
from scipy import special
from scipy.optimize import least_squares
def bern_eq(x, f, q):
    return np.concatenate([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])
    
def FF_Bernoulli(F, G, delta, flow, n, T0, TActual, eps):
    # F = F_bern; G = G_bern; delta = delta_bern; 
    # Initiate: Bernoulli
    mt  = np.zeros((2,TActual))
    Ct  = np.zeros((2,2,TActual))
    at  = np.zeros((2,TActual))
    Rt  = np.zeros((2,2,TActual))
    rt  = np.zeros((1,TActual))
    st  = np.zeros((1,TActual))
    ft  = np.zeros((1,TActual+1))
    qt  = np.zeros((1,TActual+1))
    sft = np.zeros((1,TActual))
    sqt = np.zeros((1,TActual))
    skipped = np.zeros((1,TActual))
    
    # Prior: Bernoulli
    m0 = np.array([[0],
                   [0]])
    C0 = 0.1*np.eye(2)
    
    # Transform
    zt = np.array(flow[(n-1): n, :] > eps)

    # Forward filtering: Bernoulli
    for t in range(TActual):
        # t = 14
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
        xnew = least_squares(bern_eq, [1, 1],args = (ft[:, t], qt[:, t]), bounds = ((0, 0), (1e16, 1e16)))
        # xnew = np.exp(fsolve(bern_eq, [1, 1], args = (ft[:, t], qt[:, t])))
        rt[:, t] = xnew.x[0]
        st[:, t] = xnew.x[1]
        
        # Update ft, qt
        sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, T0+t]) - special.polygamma(0, st[:, t] + 1 - zt[:, T0+t])
        sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, T0+t]) + special.polygamma(1, st[:, t] + 1 - zt[:, T0+t]);
        
        # Posterior mt, Ct
        mt[:,t]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
        Ct[:,:,t] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
    
    return mt, Ct, at, Rt, rt, st, skipped

def RA_Bernoulli(TActual, F, G, mt, Ct, at, Rt, skipped):
    
    # F = F_bern; G = G_bern; mt = mt_bern; Ct = Ct_bern; at = at_bern; Rt = Rt_bern; skipped = skipped_bern;
    # Initialization
    sat = np.zeros((2,TActual))
    sRt = np.zeros((2,2,TActual))
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
        xnew = least_squares(bern_eq, [1, 1],args = (ssft[:, t], ssqt[:, t]), bounds = ((0, 0), (1e16, 1e16)))
        ssrt[:,t]  = xnew.x[0]
        ssst[:,t]  = xnew.x[1]
    
    return sat, sRt, ssrt, ssst