from scipy.linalg import block_diag
from scipy import special
import numpy as np
import numpy.matlib

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

def FF_Poisson(F, G, delta, flow, n, mN, T0, TActual, eps, RE_rho, conditional_shift):
   
    # Incorporate Random effect: change F, G, delta, mt, Ct, at, Rt
    # Add it anyway. RE_rho = 1 is the case when I don't want RE
    # Too bad matlab cannot assign default value as I do in R
    
    # F = F_pois; G = G_pois; delta = delta_pois; RE_rho = RE_rho_pois; conditional_shift = conditional_shift_pois;
    
    F = np.r_[np.array([[1]]), F]
    G = block_diag(0, G)
    delta = np.diag(np.c_[np.array([RE_rho]), np.matlib.repmat(delta,1,2)][0,:])

    # Initialize parameters
    mt  = np.zeros((3,TActual))
    Ct  = np.zeros((3,3,TActual))
    at  = np.zeros((3,TActual))
    Rt  = np.zeros((3,3,TActual))
    rt  = np.zeros((1,TActual))
    ct  = np.zeros((1,TActual))
    ft  = np.zeros((1,TActual+1))
    qt  = np.zeros((1,TActual+1))
    sft = np.zeros((1,TActual))
    sqt = np.zeros((1,TActual))
    
    skipped = np.zeros((1, TActual))

    # Prior: Poisson
    m0 = np.array([1, np.log(max(flow[n,T0-1],1)), 0]).reshape(3, 1)
    C0 = 0.1 * np.eye(3)

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
            qt[:, t]      = F[1:3].T @ Rt[1:3,1:3,t] @ F[1:3] # just for clarity
            
            # Random effect
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
            
            
            sft[:, t] = special.polygamma(0, rt[:, t]+xt[T0+t]) - np.log(ct[:, t]+1)
            # sft[:, t] = special.polygamma(0, rt[:, t]+xt[T0+t]) - np.log(ct[:, t]+mN[t:,0])
            sqt[:, t] = special.polygamma(1, rt[:, t]+xt[T0+t])
            mt[:,t] = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t])/qt[:, t]
            Ct[:,:,t] = Rt[:,:,t]-Rt[:,:,t] @ F @ (F.T) @ Rt[:,:,t] * (1-sqt[:, t]/qt[:, t])/qt[:, t]
    return mt, Ct, at, Rt, rt, ct, skipped



def RA_Poisson(TActual, F, G, mt, Ct, at, Rt, skipped):
    
   
    # F = F_pois; G = G_pois; mt = mt_pois; Ct = Ct_pois; at = at_pois; Rt = Rt_pois; skipped = skipped_pois;
   

    # Change F, G and delta for random effect
    F = np.r_[np.array([[1]]), F]
    G = block_diag(0, G)

    # Initialization
    sat = np.zeros((3,TActual))
    sRt = np.zeros((3,3,TActual))
    ssft = np.zeros((1,TActual))
    ssqt = np.zeros((1,TActual))
    ssrt = np.zeros((1,TActual))
    ssct = np.zeros((1,TActual))
   
    # Define the last time point
    sat[:,TActual-1]   = mt[:,TActual-1]
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
        
        #sim_sat[:,t] = mt[:,t] + Bt @ (sim_at[:,t+1] - at[:,t+1])
        #sim_sRt[:,:,t] = Ct[:,:,t] - Bt @ (sim_Rt[:,:,t+1]) @ Bt.T
    
    # Approximate rt, ct
    # for t in np.arange(TActual-2, -1, -1):
    for t in np.arange(-1, -TActual-1, -1):
        ssft[:,t]     = F.T @ sat[:,t]
        ssqt[:,t]     = F.T @ sRt[:,:,t] @ F
        # fprintf('t = %d, ssft = %.2f, ssqt = %.2f\n', t, ssft(t), ssqt(t));
        
        # Get numerical root of Gamma approximation
# =============================================================================
#         rnew = least_squares(pois_eq, 1, args = (ssqt[:, t]), bounds = ((0), (1e16)))
#         ssrt[:, t] = rnew.x
#         ssct[:, t] = np.exp(special.polygamma(0, ssrt[:, t])-ssft[:,t])
# =============================================================================
        ssrt[:, t] = pois_eq_solver(1/ssqt[0, t], ssqt[0, t])
        ssct[:, t] = np.exp(special.polygamma(0, ssrt[:, t])-ssft[:, t])
    
    return sat, sRt, ssrt, ssct









