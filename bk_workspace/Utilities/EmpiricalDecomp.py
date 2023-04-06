import numpy as np
import numpy.random

def EmpiricalDecomp(flow, flowIndex, categories, m, N, I, TActual, T0, eps):
    # Initialization
    mEf = np.zeros((1, TActual))
    mEai = np.zeros((I, TActual))
    mEbj = np.zeros((I, TActual))
    mEgij = np.zeros((N, TActual))    
    modifiedFlow = np.zeros((N, TActual))
    
    # Adjust flow
    for n in range(N):
        #i = flowIndex(n,1);
        #mN = m(:, categories==i);
        # Lower bound = 1 on emprical values, consistent with that for samples
        #modifiedFlow(n,:) = reshape(log(max(flow(n, (T0+1):(T0+TActual)), 1)),...
        #    [1,TActual]) - reshape(log(max(mN,eps)), [1,TActual]);
        
        np.log(np.clip(flow[n, T0:(T0+TActual)], 1, 1e16)).reshape(1,TActual)
            
        modifiedFlow[n, :] = \
            np.log(np.clip(flow[n, T0:(T0+TActual)], 1, 1e16)).reshape(1,TActual)
    
    modifiedFlow[flow[:, T0:(T0+TActual)] == eps] = 0
    modifiedFlow[modifiedFlow < 0] = 0
    # Empirical a_it, b_jt
    for t in range(TActual):
        # t = 0
        mEf[0, t] = np.sum(modifiedFlow[:, t]) / I**2
        
        for i in range(I):
            mEai[i, t] = np.sum(modifiedFlow[(flowIndex[:, 0] == categories[i]), t]) / I - mEf[0, t]
            mEbj[i, t] = np.sum(modifiedFlow[(flowIndex[:, 1] == categories[i]), t]) / I - mEf[0, t]
        
    
    # Empirical g_ijt
    for t in range(TActual):
        # t = 0
        for n in range(N):
            # n = 0
            i = np.where(categories==flowIndex[n, 0])[0]
            j = np.where(categories==flowIndex[n, 1])[0]
            mEgij[n, t] = \
                modifiedFlow[n, t] - mEf[0, t] - mEai[i, t] - mEbj[j, t]
    
    return mEf, mEai, mEbj, mEgij
        
    

