import numpy as np
import numpy.random
def Recouple_DGM2(rt_bern_all, st_bern_all, rt_pois_all, st_pois_all, \
    conditional_shift_pois, \
    TActual, flowIndex, categories, sampleSize, I, N):
    # sampleSize = 1000
    
    # Sample
    mf = np.zeros((sampleSize,1))
    mai = np.zeros((sampleSize,I))
    mbj = np.zeros((sampleSize,I))
    mgij = np.zeros((sampleSize,N))
    
    # Sample mean, upper, lower bounds
    # f
    fEst = np.zeros((1, TActual))
    fUpper = np.zeros((1,TActual))
    fLower = np.zeros((1,TActual))
    
    # ai
    aiEst = np.zeros((I, TActual))
    aiUpper = np.zeros((I, TActual))
    aiLower = np.zeros((I, TActual))
    
    # bj
    bjEst = np.zeros((I, TActual))
    bjUpper = np.zeros((I, TActual))
    bjLower = np.zeros((I, TActual))
    
    # g_ij
    gijEst = np.zeros((N, TActual))
    gijUpper = np.zeros((N, TActual))
    gijLower = np.zeros((N, TActual))
    
    for t in range(TActual):
        print("DGM:",t+1, "/", TActual, sep=(""))
        # t = 0
        # Get sample of phi
        bsSample = np.zeros((N, sampleSize))
# =============================================================================
#         for n = 1:N
#             p_samp = betarnd(rt_bern_all(t, n), st_bern_all(t, n), 1, sampleSize);
#             lambda_samp = betarnd(rt_pois_all(t, n), st_pois_all(t, n), 1, sampleSize);
#             bsSample(n, :) = p_samp .* (1 + lambda_samp);
#         end
#         bsLogSample = log(max(bsSample, 1));
# =============================================================================

        # Fitted y as phi
        for n in range(N):
            # n = 0
            p_bern = rt_bern_all[t, n] / (rt_bern_all[t, n] + st_bern_all[t, n])
            
            rnd_bern = np.random.binomial(1, p_bern, sampleSize)
            
            # Sample from Negative Binomial (Poisson part)
            p_nb   = st_pois_all[t, n]/(st_pois_all[t, n] + 1)
            k_nb   = rt_pois_all[t, n]
            rnd_nb = np.random.negative_binomial(k_nb, p_nb, sampleSize)

            # combine
            bsSample[n, :] = rnd_bern * (conditional_shift_pois + rnd_nb)          
        
        
        bsLogSample = np.log(np.clip(bsSample, 1, 1e16))
        
        # f_t, a_it, b_jt--------------       
        # Get sample
        for k in range(sampleSize):
            # k = 0
            mf[k, 0] = np.sum(bsLogSample[:, k]) / I**2
            for i in range(I):
                # i = 0
                find_i = flowIndex[:, 0] == categories[i]
                find_j = flowIndex[:, 1] == categories[i]
                mai[k, i] = np.sum(bsLogSample[find_i, k]) / I - mf[k, 0]
                mbj[k, i] = np.sum(bsLogSample[find_j, k]) / I - mf[k, 0]
            
        
        
        # Get sample means, upper and lower bounds (colMeans)
        fEst[0, t] = np.mean(mf[:, 0])
        fUpper[0, t]   = np.quantile(mf[:, 0], 0.975)
        fLower[0, t]   = np.quantile(mf[:, 0], 0.025)
        
        aiEst[:, t]   = np.mean(mai, axis= 0)
        aiUpper[:, t] = np.quantile(mai, 0.975, axis = 0)
        aiLower[:, t] = np.quantile(mai, 0.025, axis = 0)
        
        bjEst[:, t]   = np.mean(mbj, axis= 0)
        bjUpper[:, t] = np.quantile(mbj, 0.975, axis = 0)
        bjLower[:, t] = np.quantile(mbj, 0.025, axis = 0)
        
        # g_ijt --------------------
        # Get sample
        for n in range(N):   
            # n = 10             
            i = categories == flowIndex[n, 0]
            j = categories == flowIndex[n, 1]
            mgij[:, n] = \
                np.log(np.clip(bsSample[n, :], 1, 1e16)).reshape(sampleSize, )-\
                (mai[:, i]+mbj[:, j]).reshape(sampleSize, )-\
                (mf[:, 0]).reshape(sampleSize, )
         
        
        # Get sample means, upper and lower bounds
        
        gijEst[:, t] = np.mean(mgij, axis= 0)
        gijUpper[:, t] = np.quantile(mgij, 0.975, axis = 0)
        gijLower[:, t] = np.quantile(mgij, 0.025, axis = 0)
            

    # Values in original scale
    # fEst_exp = zeros(1,TActual);
    # aiEst_exp = zeros(I,TActual);
    # bjEst_exp = zeros(I,TActual);
    # gijEst_exp = zeros(N,TActual);
    # for t = 1:TActual
    #     fEst_exp(t) = mean(exp(mf(:,t)));
    #     aiEst_exp(:,t) = mean(exp(mai(:,:,t)),1);
    #     bjEst_exp(:,t) = mean(exp(mbj(:,:,t)),1);
    #     gijEst_exp(:,t) = mean(exp(mgij(:,:,t)),1);
        
    #     disp(strcat('DGM Exp:', num2str(t), '/', num2str(TActual)))
    # end
    
    return fEst, fUpper, fLower, aiEst, aiUpper, aiLower, bjEst, bjUpper, bjLower, \
        gijEst, gijUpper, gijLower

