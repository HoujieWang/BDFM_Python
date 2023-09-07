import numpy as np
import scipy
from scipy import special
from scipy.stats import multivariate_normal
from scipy.special import polygamma
from scipy.optimize import least_squares

def dirichlet_eq(x, mu, sigma2, cov):
    return  np.concatenate((polygamma(0, x[1:]) - polygamma(0, x[0]) - mu, \
                    polygamma(1, x[1:]) + polygamma(1, x[0]) - sigma2,\
                    np.array([polygamma(1, x[0]) - cov])))
        
def bern_eq(x, f, q):
    # x = np.array([1, 1]); f = all_f[i]; q = all_q[i]
    return np.array([special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f, 
special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q])

def fun1(x, f, q):
    return abs(special.polygamma(0, x[0]) - special.polygamma(0, x[1])-f) + \
        abs(special.polygamma(1, x[0]) + special.polygamma(1, x[1])-q)

def bern_eq_solver(x, f, q):
    # x = np.array([[1/qt[0, t]], [1/qt[0, t]]]); f = ft[0, t]; q = qt[0, t]
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

def bern_eq_solver2(x, f, q):
    # x = np.array([[1/qt[0, t]], [1/qt[0, t]]]); f = ft[0, t]; q = qt[0, t]
    x_exp = np.exp(x) # initial input
    f_org = np.array([[special.polygamma(0, x_exp[0,0]) - special.polygamma(0, x_exp[1,0])-f], 
                      [special.polygamma(1, x_exp[0,0]) + special.polygamma(1, x_exp[1,0])-q]])
    Df = np.array([[special.polygamma(1, x_exp[0,0])*x_exp[0,0], -special.polygamma(1, x_exp[1,0])*x_exp[1,0]],
                   [special.polygamma(2, x_exp[0,0])*x_exp[0,0], special.polygamma(2, x_exp[1,0])*x_exp[1,0]]])
    
    x_new = x - np.linalg.inv(Df) @ f_org
    while (abs(x[0]/x_new[0]-1)>=1e-5 and abs(x[1]/x_new[1]-1)>=1e-5):
            x = x_new
            x_exp = np.exp(x) # initial input
            f_org = np.array([[special.polygamma(0, x_exp[0,0]) - special.polygamma(0, x_exp[1,0])-f], 
                              [special.polygamma(1, x_exp[0,0]) + special.polygamma(1, x_exp[1,0])-q]])
            Df = np.array([[special.polygamma(1, x_exp[0,0])*x_exp[0,0], -special.polygamma(1, x_exp[1,0])*x_exp[1,0]],
                           [special.polygamma(2, x_exp[0,0])*x_exp[0,0], special.polygamma(2, x_exp[1,0])*x_exp[1,0]]])
            
            x_new = x - np.linalg.inv(Df) @ f_org
            print(x_new)
    x_new = np.exp(x_new)
    return x_new[0, 0], x_new[1, 0]


# def FF_Bernoulli2(F, G, delta, zt, pr_prob):

#     # F = F_bern; G = G_bern; delta = delta_bern
#     # zt = zt[0:1,:]
#     # pr_prob = 0.5
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
    
#     a0 = np.append(scipy.special.logit(pr_prob), np.zeros((F.shape[0]-1, )))
    
#     mt[:, 0] = np.linalg.inv(G) @ a0
#     Ct[:, :, 0] = 0.1*np.eye(d2)
#     # Forward filtering: Bernoulli
#     count = 0
#     for t in range(TActual):
#         if (zt[:,t] == 0) or (zt[:,t] == 1): # If we have data for time t:
#             count = 0
#             at[:,t]   = G @ mt[:,t]
            
#             Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
#             Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            

#             ft[:, t]     = F.T @ at[:,t]
#             qt[:, t]     = F.T @ Rt[:,:,t] @ F
            
#             xnew = least_squares(bern_eq, [1, 1], \
#                                   args = (ft[0, t], qt[0, t]), \
#                                   bounds = ((0, 0), (float("inf"), float("inf"))))
#             rt[:,t]  = xnew.x[0]
#             st[:,t]  = xnew.x[1]
            
#             # Update ft, qt
#             sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, t]) - special.polygamma(0, st[:, t] + 1 - zt[:, t])
#             sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, t]) + special.polygamma(1, st[:, t] + 1 - zt[:, t]);
            
#             # Posterior mt, Ct
#             mt[:,t+1]   = at[:,t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
#             Ct[:,:,t+1] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
            
#         else: 
#             count+=1
#             at[:,t]   = G @ mt[:,t]
#             if (count < 3):
#                 Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
#                 Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
#             else:
#                 Rt[:,:,t] = Rt[:,:,t-1]
                
#             mt[:,t+1] = at[:,t] 
#             Ct[:,:,t+1] = Rt[:,:,t]
#             rt[:, t] = rt[:, t-1]
#             st[:, t] = st[:, t-1]
#             skipped[:, t] = True
            
#     return mt[:, 1:], Ct[:,:,1:], at, Rt, rt, st, skipped

def FF_Bernoulli3(F, G, delta, zt, pr_prob):

    # F = F_bern; G = G_bern; delta = delta_bern
    # zt = flow_count_i[:, :672]
    # TActual = 672
    # pr_prob = 0.00001
    # Initiate: Bernoulli
    npath = zt.shape[0]
    d1, d2 = G.shape
    TActual = zt.shape[1]
    mt  = np.zeros((d2,npath, TActual+1))
    Ct = [None]*(TActual+1)
    # Ct  = np.zeros((d2,d2,TActual+1))
    
    at  = np.zeros((d1, npath, TActual))
    # Rt  = np.zeros((d1,d1,TActual))
    Rt = [None]*TActual
    
    rt  = np.zeros((npath,TActual))
    st  = np.zeros((npath,TActual))
    ft  = np.zeros((npath,TActual))
    qt  = np.zeros((npath,TActual))
    sft = np.zeros((npath,TActual))
    sqt = np.zeros((npath,TActual))
    skipped = np.zeros((npath,TActual))
    
    # Prior: Bernoulli
    
    a0 = np.append(scipy.special.logit(pr_prob), \
                   np.zeros((F.shape[0]-1, ))).reshape(d1, 1)
    
    mt[:, :, 0] = np.linalg.inv(G) @ a0 @ np.ones((1, npath))
    Ct[0] = np.hstack([0.01*np.eye(d2) for i in range(npath)])
    
    # Ct[:, :, :, 0] = 0.1*np.eye(d2)
    
    
    # Forward filtering: Bernoulli
    for t in range(TActual):
        # t = 0
        if (t%100 == 0): print(t)
        # Get the prior for mt, Ct
        at[:, :, t]   = G @ mt[:, :, t]
        Rt[t] = np.hstack([(G @ Ct[t][:, i*d1: (i+1)*d1] @ G.T) * (1+delta) \
                           for i in range(npath)])
        
        # Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
        # Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
        
            
            
        # ft[:, t]     = F.T @ at[:,t]
        ft[:, t]     = F.T @ at[:, :, t]
                
        # qt[:, t]     = F.T @ Rt[:,:,t] @ F
        qt[:, t]     = np.array([(F.T @ Rt[t][:, i*d1: (i+1)*d1] @ F)[0][0]\
                  for i in range(npath)])
        
        # rt[0, t], st[0, t] = bern_eq_solver(np.array([[1], [1]]),
        #                 ft[0, t], qt[0, t])
        xnew = np.array([least_squares(bern_eq, [1, 1], \
                       args = (ft[i, t], qt[i, t]), \
                       bounds = ((0, 0), (1e16, 1e16))).x\
                       for i in range(npath)])
        # for i in range(npath):
        #     least_squares(bern_eq, [1, 1], \
        #                    args = (ft[i, t], qt[i, t]), \
        #                    bounds = ((0, 0), (1e16, 1e16))).x
        # xnew = least_squares(bern_eq, [1, 1], \
        #                       args = (ft[0, t], qt[0, t]), \
        #                       bounds = ((0, 0), (1e16, 1e16)))
        # rt[:,t]  = xnew.x[0]
        # st[:,t]  = xnew.x[1]
        rt[:,t]  = xnew[:, 0]
        st[:,t]  = xnew[:, 1]
        
        # Update ft, qt
        sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, t]) - \
            special.polygamma(0, st[:, t] + 1 - zt[:, t])
        sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, t]) + \
            special.polygamma(1, st[:, t] + 1 - zt[:, t])
        
        # Posterior mt, Ct
        mt[:, :, (t+1)] =  np.hstack( [at[:,i:(i+1),t] + \
                     Rt[t][:, i*d1: (i+1)*d1] @ F * (sft[i, t]-ft[i, t]) / qt[i, t] \
                             for i in range(npath)])
        Ct[t+1] = np.hstack([Rt[t][:,i*d1:(i+1)*d1] - \
                             Rt[t][:,i*d1:(i+1)*d1] @ F @ F.T @ Rt[t][:,i*d1:(i+1)*d1] * \
                                 (1 - sqt[i,t] / qt[i,t]) / qt[i,t]\
                   for i in range(npath)])
        # mt[:, :, (t+1)]   = at[:,:, t] + Rt[:,:,t] @ F @ (sft[:, t]-ft[:, t]) / qt[:, t]
        # Ct[:,:,t+1] = Rt[:,:,t] - Rt[:,:,t] @ F @ F.T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
        
        
    return mt, Ct, at, Rt, rt, st, skipped
        
        
def FF_Bernoulli2(F, G, delta, zt, pr_prob):

    # F = F_bern; G = G_bern; delta = delta_bern
    # zt = zt[0:1,:]
    # pr_prob = 0.5
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
    
    a0 = np.append(scipy.special.logit(pr_prob), np.zeros((F.shape[0]-1, )))
    
    mt[:, 0] = np.linalg.inv(G) @ a0
    Ct[:, :, 0] = 0.1*np.eye(d2)
    # Forward filtering: Bernoulli
    count = 0
    for t in range(TActual):
        if (zt[:,t] == 0) or (zt[:,t] == 1): # If we have data for time t:
            count = 0
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
            sft[:, t] = special.polygamma(0, rt[:, t] + zt[:, t]) - special.polygamma(0, st[:, t] + 1 - zt[:, t])
            sqt[:, t] = special.polygamma(1, rt[:, t] + zt[:, t]) + special.polygamma(1, st[:, t] + 1 - zt[:, t]);
            
            # Posterior mt, Ct
            mt[:,t+1]   = at[:,t] + Rt[:,:,t] @ F[:,t:(t+1)] @ (sft[:, t]-ft[:, t]) / qt[:, t]
            Ct[:,:,t+1] = Rt[:,:,t] - Rt[:,:,t] @ F[:,t:(t+1)] @ F[:,t:(t+1)].T @ Rt[:,:,t] * (1 - sqt[:,t] / qt[:,t]) / qt[:,t]
            
        else: 
            count+=1
            at[:,t]   = G @ mt[:,t]
            if (count < 3):
                Rt[:,:,t] = G @ Ct[:,:,t] @ G.T
                Rt[:,:,t] = Rt[:,:,t] + Rt[:,:,t] * delta
            else:
                Rt[:,:,t] = Rt[:,:,t-1]
                
            mt[:,t+1] = at[:,t] 
            Ct[:,:,t+1] = Rt[:,:,t]
            rt[:, t] = rt[:, t-1]
            st[:, t] = st[:, t-1]
            skipped[:, t] = True
            
    return mt[:, 1:], Ct[:,:,1:], at, Rt, rt, st, skipped
        
        
        
        
# def Retro_sampling3(TActual, discount, F, G, mt, Ct, at, Rt, nSample, family):
#     # mt
#     # TActual = zt.shape[1]; discount = 0.99
#     # F=F_bern; G=G_bern; mt = mt; Ct = Ct;
#     # at = at; Rt = Rt; family = "bernoulli"
   
#     RA_samples = np.zeros((nSample, TActual))
#     # Starting Seed
#     d = G.shape[0]
#     mt_star = mt[: ,(TActual - 1):TActual] @ np.ones((1, nSample))
#     Ct_star = Ct[TActual - 1, :,:]
#     all_states = np.random.multivariate_normal(mean=np.zeros(d,), cov=Ct_star, \
#                      size=nSample).T + mt_star
#     if (family == "poisson"):
#         RA_samples[:, -1] = np.exp(F[:,(TActual-1):TActual].T @ all_states)
#     if (family == "bernoulli"):
#         RA_samples[:, -1] = scipy.special.expit(F[:,(TActual-1):TActual].T @ all_states)
        
#     # Trajectory
#     for t in np.arange(TActual-2, -1, -1):
#         # t = 
#         Rt[t+1:,:] = 0.5*(Rt[t+1,:,:] + Rt[t+1,:,:].T)
        
#         mt_star = (1 - discount) * mt[:, t:(t+1)] @ np.ones((1, nSample)) + \
#             discount * np.linalg.inv(G) @ all_states
#         Ct_star = (1 - discount) * Ct[t, :, :]
                   
#         Ct_star = 0.5*(Ct_star + Ct_star.T)
         
        
#         all_states = np.random.multivariate_normal(mean = np.zeros(d,), \
#                                       cov = Ct_star, \
#                                           size = nSample).T + np.real(mt_star)
        
#         if (family == "poisson"):
#             RA_samples[:, t] = np.exp(F[:,t:(t+1)].T @ all_states)
#         if (family == "bernoulli"):
#             RA_samples[:, t] = scipy.special.expit(F[:,t:(t+1)].T @ all_states)
#     return(RA_samples)
        
        
        
        
        
        
        
        
        
        
        
        
    
