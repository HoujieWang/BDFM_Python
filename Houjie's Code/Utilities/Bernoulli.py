import numpy as np
import scipy
from scipy import special
from scipy.stats import multivariate_normal
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

    # F = F_bern; G = G_bern; delta = delta_bern[0,0]; 
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
            # Rt[:,:,t] = G @ C0 @ G.T @ np.linalg.inv(delta)
        else:
            at[:,t]   = (G @ mt)[:,t-1]
            Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T/delta
            # Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T @ np.linalg.inv(delta)
        
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
        # qt[:, t]     = Rt[0,0,t] * F.T @ F
        qt[:, t]     = F.T @ Rt[:,:,t] @ F
        
        rt[0, t], st[0, t] = bern_eq_solver(np.array([[1/qt[0, t]], [1/qt[0, t]]]),
                       ft[0, t], qt[0, t])
        
        # Update ft, qt
        sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, T0+t]) - special.polygamma(0, st[:, t] + 1 - zt[:, T0+t])
        sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, T0+t]) + special.polygamma(1, st[:, t] + 1 - zt[:, T0+t]);
        
        # Posterior mt, Ct
        mt[:,t]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
        Ct[:,:,t] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
        # np.fill_diagonal(Ct[:,:,t], np.clip(Ct[:,:,t].diagonal(), 1e-10, 1e16))
    return mt, Ct, at, Rt, rt, st, skipped

def FF_Bernoulli2(F, G, delta, flow, n, T0, TActual, eps):
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
            # Rt[:,:,t] = G @ C0 @ G.T/delta
            
            Rt[:,:,t] = G @ C0 @ G.T
            Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            
        else:
            at[:,t]   = (G @ mt)[:,t-1]
            # Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T/delta
            Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T
            Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            
        
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
        # qt[:, t]     = Rt[0,0,t] * F.T @ F
        qt[:, t]     = F.T @ Rt[:,:,t] @ F
        
        rt[0, t], st[0, t] = bern_eq_solver(np.array([[1/qt[0, t]], [1/qt[0, t]]]),
                       ft[0, t], qt[0, t])
        
        # Update ft, qt
        sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, T0+t]) - special.polygamma(0, st[:, t] + 1 - zt[:, T0+t])
        sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, T0+t]) + special.polygamma(1, st[:, t] + 1 - zt[:, T0+t]);
        
        # Posterior mt, Ct
        mt[:,t]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
        Ct[:,:,t] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
        Ct[:,:,t] = 0.5*(Ct[:,:,t]+Ct[:,:,t].T)
        # np.fill_diagonal(Ct[:,:,t], np.clip(Ct[:,:,t].diagonal(), 1e-10, 1e16))
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
        # t = 390
        # Skipp the time points not updated
        if skipped[:,t+1]:
            sat[:,t] = sat[:,t+1]
            sRt[:,:,t] = sRt[:,:,t+1]
            continue
        
        temp = np.linalg.eig(Rt[:,:,t+1])
        Bt = Ct[:,:,t] @ G.T @ temp[1] @ np.diag(1/temp[0]) @ [1].T
        # Bt = Ct[:,:,t] @ G.T @ scipy.linalg.inv(Rt[:,:,t+1])
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

def Retro_sampling(TActual, discount, F, G, mt, Ct, at, Rt, skipped, nSample, family):
    # F=F_bern; G=G_bern; mt = mt_bern; Ct = Ct_bern; discount =np.array([0.95])
    # at = at_bern; Rt = Rt_bern; skipped = skipped_pois; family = "bernoulli"
   
    RA_samples = np.zeros((nSample, TActual))
    
    # Starting Seed
    d = G.shape[0]
    all_states = np.random.multivariate_normal(mean=mt[: ,TActual - 1], \
                     cov=Ct[:,: ,TActual - 1], \
                     size=nSample).T
    # all_m = np.zeros((d, TActual))
    # Trajectory
    # for t in np.arange(TActual-2, -1, -1):
    for t in np.arange(TActual-2, -1, -1):
        # t = 
        Rt[:,:,t+1] = 0.5*(Rt[:,:,t+1] + Rt[:,:,t+1].T)
        
        if discount.shape[0] == 1:
            mt_star = (1 - discount[0]) * mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
                discount[0] * np.linalg.inv(G) @ all_states
            Ct_star = (1 - discount[0]) * Ct[:, :, t]
        else:
            if (scipy.linalg.det(Rt[:,:,t+1]) > 0):
                Bt = Ct[:,:,t] @ G.T @ scipy.linalg.inv(Rt[:,:,t+1])
                
            else:
                temp = scipy.linalg.eig(Rt[:,:,t+1])
                Bt = Ct[:,:,t] @ G.T @temp[1] @ np.diag(1 / np.real(np.sqrt(temp[0]))) @ temp[1].T
            mt_star = mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
                       Bt @ (all_states - at[:,  t: (t + 1)] @ np.ones((1, nSample)))
            Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T
                   
        Ct_star = 0.5*(Ct_star + Ct_star.T)
         
        
        all_states = np.random.multivariate_normal(mean = np.zeros(d,), \
                                      cov = Ct_star, \
                                          size = nSample).T + np.real(mt_star)
        
        if (family == "poisson"):
            RA_samples[:, t] = np.exp(F.T @ all_states)
        if (family == "bernoulli"):
            RA_samples[:, t] = scipy.special.expit(F.T @ all_states)
    return(RA_samples)

def Retro_sampling2(TActual, F, G, mt, Ct, at, Rt, skipped, nSample, family):
    # F=F_pois; G=G_pois; mt = mt_pois; Ct = Ct_pois; discount =np.array([0.9, 0.95])
    # at = at_pois; Rt = Rt_pois; skipped = skipped_pois; family = "poisson"
    # nSample = 2
    RA_samples = np.zeros((nSample, TActual))
    # Starting Seed
    d = G.shape[0]
    mt_star = mt[: ,(TActual - 1):TActual] @ np.ones((1, nSample))
    Ct_star = Ct[:,: ,TActual - 1]
    all_states = np.random.multivariate_normal(mean=np.zeros(d,), cov=Ct_star, \
                     size=nSample).T + mt_star
    if (family == "poisson"):
        RA_samples[:, -1] = np.exp(F.T @ all_states)
    if (family == "bernoulli"):
        RA_samples[:, -1] = scipy.special.expit(F.T @ all_states)
        
    # Trajectory
    for t in np.arange(TActual-2, -1, -1):
        # t = 548
        
        if skipped[:,t+1] == 0:
            Bt = Ct[:,:,t] @ G.T @ scipy.linalg.inv(Rt[:,:,t+1])
            mt_star = mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
                       Bt @ (all_states - at[:,  t: (t + 1)] @ np.ones((1, nSample)))
            Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T

                   
        Ct_star = 0.5*(Ct_star + Ct_star.T)
                 
        
        
        all_states = np.random.multivariate_normal(mean = np.zeros(d,), \
                                      cov = Ct_star, \
                                          size = nSample).T + mt_star

        if (family == "poisson"):
            RA_samples[:, t] = np.exp(F.T @ all_states)
        if (family == "bernoulli"):
            RA_samples[:, t] = scipy.special.expit(F.T @ all_states)
            
            
    return(RA_samples)
