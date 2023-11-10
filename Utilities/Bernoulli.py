import numpy as np
import scipy
from scipy import special
from scipy.special import polygamma
from scipy.optimize import least_squares, fsolve
# def bern_eq(x, f, q):
#     return np.concatenate([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
# special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])

def bern_eq(x, f, q):
    # x = np.array([1, 1]); f = all_f[i]; q = all_q[i]
    return np.array([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])

def bern_eq2(x, f, q):
    # x = np.array([1, 1]); f = all_f[i]; q = all_q[i]
    x = exp(x)
    return np.array([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
                     special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])

def dirichlet_eq(x, mu, sigma2, cov):
    return  np.concatenate((polygamma(0, x[1:]) - polygamma(0, x[0]) - mu, \
                    polygamma(1, x[1:]) + polygamma(1, x[0]) - sigma2,\
                    np.array([polygamma(1, x[0]) - cov])))
        
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

# def FF_Bernoulli2(F, G, delta, flow, n, T0, TActual, eps):
#     # F = F_bern; G = G_bern; delta = delta_bern; 
#     # flow = flow_count
#     # Initiate: Bernoulli
#     d1, d2 = G.shape
#     mt  = np.zeros((d2,TActual))
#     Ct  = np.zeros((d2,d2,TActual))
#     at  = np.zeros((d1,TActual))
#     Rt  = np.zeros((d1,d1,TActual))
#     rt  = np.zeros((1,TActual))
#     st  = np.zeros((1,TActual))
#     ft  = np.zeros((1,TActual+1))
#     qt  = np.zeros((1,TActual+1))
#     sft = np.zeros((1,TActual))
#     sqt = np.zeros((1,TActual))
#     skipped = np.zeros((1,TActual))
    
#     # Prior: Bernoulli
#     # m0 = np.array([[0],
#     #                [0]])
#     # C0 = 0.1*np.eye(2)
    
#     m0 = np.zeros((d2, 1))
#     C0 = 0.1*np.eye(d2)
    
#     # Transform
#     zt = np.array(flow[n: (n+1), :] > eps)

#     # Forward filtering: Bernoulli
#     for t in range(TActual):
#         # t = 0
#         # When t = 1. Get prior from m0, C0
#         if t == 0: 
#             at[:, t]   = (G @ m0)[:,0]
#             # Rt[:,:,t] = G @ C0 @ G.T/delta
            
#             Rt[:,:,t] = G @ C0 @ G.T
#             Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            
#         else:
#             at[:,t]   = (G @ mt)[:,t-1]
#             # Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T/delta
#             Rt[:,:,t] = G @ Ct[:,:,t-1] @ G.T
#             Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            
        
#         # Stop updating when same value keeps coming back (i.e. all 1)
#         # To visit: is this stopping rule right? Work for all 0?
        
#         if t > 0 and abs(rt[:,t-1] / (rt[:,t-1] + st[:,t-1]) - zt[:,t]) < 1e-2:
#             at[:,t] = at[:,t-1]
#             Rt[:,:,t] = Rt[:,:,t-1]
#             mt[:,t] = mt[:,t-1]
#             Ct[:,:,t] = Ct[:,:,t-1]
#             rt[:, t] = rt[:, t-1]
#             st[:, t] = st[:, t-1]
#             skipped[:, t] = True
#             continue
        
#         ft[:, t]     = F.T @ at[:,t]
#         # qt[:, t]     = Rt[0,0,t] * F.T @ F
#         qt[:, t]     = F.T @ Rt[:,:,t] @ F
        
#         rt[0, t], st[0, t] = bern_eq_solver(np.array([[1/qt[0, t]], [1/qt[0, t]]]),
#                        ft[0, t], qt[0, t])
        
#         # Update ft, qt
#         sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, T0+t]) - special.polygamma(0, st[:, t] + 1 - zt[:, T0+t])
#         sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, T0+t]) + special.polygamma(1, st[:, t] + 1 - zt[:, T0+t]);
        
#         # Posterior mt, Ct
#         mt[:,t]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
#         Ct[:,:,t] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
#         Ct[:,:,t] = 0.5*(Ct[:,:,t]+Ct[:,:,t].T)
#         # np.fill_diagonal(Ct[:,:,t], np.clip(Ct[:,:,t].diagonal(), 1e-10, 1e16))
#     return mt, Ct, at, Rt, rt, st, skipped




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

def Retro_sampling(TActual, F, G, mt, Ct, at, Rt, nSample, family, discount=[]):
    # TActual = at.shape[1]; discount = 0.995
    # F=F_bern; G=G_bern; mt = mt; Ct = Ct;
    # at = at; Rt = Rt; family = "none"
    # marginal = True
    RA_samples = np.zeros((nSample, TActual))
    # Starting Seed
    d = G.shape[0]
    mt_star = mt[: ,(TActual - 1):TActual] @ np.ones((1, nSample))
    Ct_star = Ct[:,: ,TActual - 1]
    all_states = np.random.multivariate_normal(mean=np.zeros(d,), cov=Ct_star, \
                     size=nSample).T + mt_star
    state_mu = np.zeros((mt_star.shape[0], TActual))
    state_var = np.zeros(Ct.shape)
    lambda_mu = np.zeros((TActual, ))
    lambda_var = np.zeros((TActual, ))
    
    state_mu[:, TActual-1] = mt[:, TActual-1]
    state_var[:,:, TActual-1] = Ct[:,: ,TActual-1]
    lambda_mu[TActual-1] = F[:,(TActual-1):TActual].T @ state_mu[:, TActual-1]
    lambda_var[TActual-1] = F[:,(TActual-1):TActual].T @ state_var[:,:, TActual-1] @ F[:,(TActual-1):TActual]
    
    
    if (family == "poisson"):
        RA_samples[:, -1] = np.exp(F[:,(TActual-1):TActual].T @ all_states)
    if (family == "bernoulli"):
        RA_samples[:, -1] = scipy.special.expit(F[:,(TActual-1):TActual].T @ all_states)
    if (family == "none"):
        RA_samples[:, -1] = F[:,(TActual-1):TActual].T @ all_states
    # Trajectory
    for t in np.arange(TActual-2, -1, -1):
        # t = TActual-2
        # Rt[:,:,t+1] = 0.5*(Rt[:,:,t+1] + Rt[:,:,t+1].T)
        # G_inv = np.linalg.inv(G)
        # mt_star = (1 - discount) * mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
        #     discount * G_inv @ all_states
        # Ct_star = (1 - discount) * Ct[:, :, t]
        Bt = Ct[:,:,t] @ G.T @ scipy.linalg.inv(Rt[:,:,t+1])
        mt_star = mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
                    Bt @ (all_states - at[:,  (t+1): (t+2)] @ np.ones((1, nSample)))
        Ct_star = Ct[:, :, t] - Bt @ Rt[:, :, t + 1] @ Bt.T  
                 
        Ct_star = 0.5*(Ct_star + Ct_star.T)
         
        
        all_states = np.random.multivariate_normal(mean = np.zeros(d,), \
                                      cov = Ct_star, \
                                          size = nSample).T + np.real(mt_star)
            
        # state_mu[:, t] = (1 - discount) * mt[:, t] + \
        #     discount * G_inv @ state_mu[:, (t+1)]
        # state_var[:,:, t] = discount**2 * G_inv @ state_var[:,:, (t+1)] @ G_inv.T + \
        #     (1 - discount) * Ct[:, :, t]
            
        state_mu[:, t:(t+1)] = mt[:, t:(t+1)] + Bt @ (state_mu[:, (t+1):(t+2)] - at[:,  (t+1):(t+2)])
        state_var[:,:, t] = Ct_star
            
        lambda_mu[t] = F[:,t:(t+1)].T @ state_mu[:, t]
        lambda_var[t] = F[:,t:(t+1)].T @ state_var[:,:, t] @ F[:,t:(t+1)]
        
        if (family == "poisson"):
            RA_samples[:, t] = np.exp(F[:,t:(t+1)].T @ all_states)
        if (family == "bernoulli"):
            RA_samples[:, t] = scipy.special.expit(F[:,t:(t+1)].T @ all_states)
        if (family == "none"):
            RA_samples[:, t] = F[:,t:(t+1)].T @ all_states
        
    return [np.array([lambda_mu, lambda_var]), RA_samples]

     
def FF_Bernoulli2(F, G, delta, zt, nt=[], pr_var = 0.1, no_data_update = False):
    if (len(nt) == 0): 
        nt=np.repeat(1, zt.shape[1])
    # F = F_ii; G = G_bern; delta = delta_bern
    # zt =trn_flow_i[[ii],:]; pr_var = pr_var
    # Initiate: Bernoulli
    d1, d2 = G.shape
    TActual = zt.shape[1]
    mt  = np.zeros((d2,TActual+1))
    Ct  = np.zeros((d2,d2,TActual+1))
    at  = np.zeros((d1,TActual))
    Rt  = np.zeros((d1,d1,TActual))
    rt  = np.zeros((1,TActual))
    st  = np.zeros((1,TActual))
    ft  = np.zeros((1,TActual))
    qt  = np.zeros((1,TActual))
    sft = np.zeros((1,TActual))
    sqt = np.zeros((1,TActual))
    skipped = np.zeros((1,TActual))
    
    # Prior: Bernoulli
    
    
    # a0 = np.concatenate([np.zeros((F.shape[0]-3, )), \
    #                       np.array([scipy.special.logit(pr_prob)]), \
    #                           np.zeros((F.shape[0]-3, ))])
    a0 = np.zeros((d2, ))
    mt[:, 0] = np.linalg.inv(G) @ a0
    Ct[:, :, 0] = pr_var*np.eye(d2)
    # Forward filtering: Bernoulli
    # count = 0
    for t in range(TActual):
        if (zt[:,t] >= 0): # If we have data for time t:
            # count = 0
            at[:,t]   = G @ mt[:,t]
            
            Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
            Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            

            ft[:, t]     = F[:,t:(t+1)].T @ at[:,t]
            qt[:, t]     = F[:,t:(t+1)].T @ Rt[:,:,t] @ F[:,t:(t+1)]
            
            xnew = least_squares(bern_eq, [1, 1], \
                                  args = (ft[0, t], qt[0, t]), \
                                  bounds = ((0, 0), (float("inf"), float("inf"))))
            rt[:,t]  = xnew.x[0]
            st[:,t]  = xnew.x[1]
            
            # Update ft, qt
            sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, t]) - special.polygamma(0, st[:, t] + nt[t] - zt[:, t])
            sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, t]) + special.polygamma(1, st[:, t] + nt[t] - zt[:, t]);
            
            # Posterior mt, Ct
            mt[:,t+1]   = at[:,t] + Rt[:,:,t] @ F[:,t:(t+1)] @ (sft[:, t]-ft[:, t]) / qt[:, t]
            Ct[:,:,t+1] = Rt[:,:,t] - Rt[:,:,t] @ F[:,t:(t+1)] @ F[:,t:(t+1)].T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
            
        else: 
            # count+=1
            at[:,t]   = G @ mt[:,t]
            Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
            
            if no_data_update:
                ft[:, t]     = F[:,t:(t+1)].T @ at[:,t]
                qt[:, t]     = F[:,t:(t+1)].T @ Rt[:,:,t] @ F[:,t:(t+1)]
                
                xnew = least_squares(bern_eq, [1, 1], \
                                      args = (ft[0, t], qt[0, t]), \
                                      bounds = ((0, 0), (float("inf"), float("inf"))))
                rt[:,t]  = xnew.x[0]
                st[:,t]  = xnew.x[1]
            else:
                rt[:, t] = rt[:, t-1]
                st[:, t] = st[:, t-1]
            
            # if (count < 2):
            #     Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
            #     Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            # else:
            #     Rt[:,:,t] = Rt[:,:,t-1]
                
            mt[:,t+1] = at[:,t] 
            Ct[:,:,t+1] = Rt[:,:,t]
            skipped[:, t] = True
            
    return mt[:, 1:], Ct[:,:,1:], at, Rt, rt, st, skipped

# def FF_Bernoulli3(F, G, delta, zt, nt=[], pr_var = 0.1, no_data_update = False):
#     if (len(nt) == 0): 
#         nt=np.repeat(1, zt.shape[1])
#     # F = F_bern; G = G_bern; delta = delta_bern
#     # zt = trn_flow_i[[ii],:]
#     # pr_prob = 0.9
#     # Initiate: Bernoulli
#     d1, d2 = G.shape
#     TActual = zt.shape[1]
#     mt  = np.zeros((d2,TActual+1))
#     Ct  = np.zeros((d2,d2,TActual+1))
#     at  = np.zeros((d1,TActual))
#     Rt  = np.zeros((d1,d1,TActual))
#     rt  = np.zeros((1,TActual))
#     st  = np.zeros((1,TActual))
#     ft  = np.zeros((1,TActual))
#     qt  = np.zeros((1,TActual))
#     sft = np.zeros((1,TActual))
#     sqt = np.zeros((1,TActual))
#     skipped = np.zeros((1,TActual))
    
#     # Prior: Bernoulli
    
    
#     # a0 = np.concatenate([np.zeros((F.shape[0]-3, )), \
#     #                       np.array([scipy.special.logit(pr_prob)]), \
#     #                           np.zeros((F.shape[0]-3, ))])
#     a0 = np.zeros((d2, ))
#     mt[:, 0] = np.linalg.inv(G) @ a0
#     Ct[:, :, 0] = pr_var*np.eye(d2)
#     # Forward filtering: Bernoulli
#     # count = 0
#     for t in range(TActual):
#         if (zt[:,t] >= 0): # If we have data for time t:
#             # count = 0
#             at[:,t]   = G @ mt[:,t]
            
#             Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
#             Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            

#             ft[:, t]     = F[:,t:(t+1)].T @ at[:,t]
#             qt[:, t]     = F[:,t:(t+1)].T @ Rt[:,:,t] @ F[:,t:(t+1)]
            
#             xnew = fsolve(bern_eq, np.array([[1],[1]]), args=(ft[0, t], qt[0, t]))
#             xnew = np.exp(xnew)
#             if (np.abs(bern_eq(xnew, ft[0, t], qt[0, t])) > 1e-2).any():
#                 sol = least_squares(bern_eq, [1, 1], \
#                                       args = (ft[0, t], qt[0, t]), \
#                                       bounds = ((0, 0), (float("inf"), float("inf"))))
                
#                 xnew = sol.x
#             rt[:,t]  = xnew[0]
#             st[:,t]  = xnew[1]
            
#             # Update ft, qt
#             sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, t]) - special.polygamma(0, st[:, t] + nt[t] - zt[:, t])
#             sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, t]) + special.polygamma(1, st[:, t] + nt[t] - zt[:, t]);
            
#             # Posterior mt, Ct
#             mt[:,t+1]   = at[:,t] + Rt[:,:,t] @ F[:,t:(t+1)] @ (sft[:, t]-ft[:, t]) / qt[:, t]
#             Ct[:,:,t+1] = Rt[:,:,t] - Rt[:,:,t] @ F[:,t:(t+1)] @ F[:,t:(t+1)].T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
            
#         else: 
#             # count+=1
#             at[:,t]   = G @ mt[:,t]
#             Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
            
#             if no_data_update:
#                 ft[:, t]     = F[:,t:(t+1)].T @ at[:,t]
#                 qt[:, t]     = F[:,t:(t+1)].T @ Rt[:,:,t] @ F[:,t:(t+1)]
                
#                 xnew = fsolve(bern_eq2, np.array([[1],[1]]), args=(ft[0, t], qt[0, t]))
#                 xnew = np.exp(xnew)
#                 if (np.abs(bern_eq(xnew, ft[0, t], qt[0, t])) > 1e-4).any():
#                     sol = least_squares(bern_eq, [1, 1], \
#                                           args = (ft[0, t], qt[0, t]), \
#                                           bounds = ((0, 0), (float("inf"), float("inf"))))
#                     xnew = sol.x
#                 rt[:,t]  = xnew[0]
#                 st[:,t]  = xnew[1]
#             else:
#                 rt[:, t] = rt[:, t-1]
#                 st[:, t] = st[:, t-1]
            
#             # if (count < 2):
#             #     Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
#             #     Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
#             # else:
#             #     Rt[:,:,t] = Rt[:,:,t-1]
                
#             mt[:,t+1] = at[:,t] 
#             Ct[:,:,t+1] = Rt[:,:,t]
#             skipped[:, t] = True
            
#     return mt[:, 1:], Ct[:,:,1:], at, Rt, rt, st, skipped


