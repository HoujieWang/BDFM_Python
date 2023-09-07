import numpy as np
import numpy.random
def Recouple_DGM2(rt_bern_all, st_bern_all, rt_pois_all, st_pois_all, \
    conditional_shift_pois, \
    TActual, flowIndex, categories, sampleSize, I, N):
    # sampleSize = 2500
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
    
    # Values in original scale
    # fEst_exp = np.zeros((1,TActual));
    # fUpper_exp = np.zeros((1,TActual))
    # fLower_exp = np.zeros((1,TActual))
    
    
    # aiEst_exp = np.zeros((I,TActual));
    # aiUpper_exp = np.zeros((I, TActual))
    # aiLower_exp = np.zeros((I, TActual))
    
    # bjEst_exp = np.zeros((I,TActual));
    # bjUpper_exp = np.zeros((I, TActual))
    # bjLower_exp = np.zeros((I, TActual))
    
    
    # gijEst_exp = np.zeros((N,TActual))
    # gijUpper_exp = np.zeros((N, TActual))
    # gijLower_exp = np.zeros((N, TActual))
    
    MC_f = np.zeros((1, TActual))
    MC_ai = np.zeros((I, TActual))
    MC_bj = np.zeros((I, TActual))
    MC_gij = np.zeros((N,TActual))
    
    # all_bsSample = np.zeros((N, sampleSize, TActual))
    # all_mf = np.zeros((sampleSize, TActual))
    # all_mai = np.zeros((sampleSize, I, TActual))
    # all_mbj = np.zeros((sampleSize, I, TActual))
    # all_mgij = np.zeros((sampleSize, N, TActual))
    
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
        
        # all_bsSample[:,:, t] = bsSample
        # all_mai[:,:, t] = mai
        # all_mbj[:,:, t] = mbj
        # all_mgij[:,:, t] = mgij
    
        # fEst_exp[0, t] = np.exp(np.mean(mf[:, 0]))
        # fUpper_exp[0, t]   = np.quantile(np.exp(mf[:, 0]), 0.975)
        # fLower_exp[0, t]   = np.quantile(np.exp(mf[:, 0]), 0.025)
        
        
        # aiEst_exp[:, t] = np.mean(np.exp(mai), axis= 0)
        # aiUpper_exp[:, t] = np.quantile(np.exp(mai), 0.975, axis = 0)
        # aiLower_exp[:, t] = np.quantile(np.exp(mai), 0.025, axis = 0)
        
        
        # bjEst_exp[:, t] = np.mean(np.exp(mbj), axis= 0)
        # bjUpper_exp[:, t] = np.quantile(np.exp(mbj), 0.975, axis = 0)
        # bjLower_exp[:, t] = np.quantile(np.exp(mbj), 0.025, axis = 0)
        
        
        
        # Get sample means, upper and lower bounds (colMeans)
        fEst[0, t]     = np.exp(np.mean(mf[:, 0]))
        fUpper[0, t]   = np.exp(np.quantile(mf[:, 0], 0.975))
        fLower[0, t]   = np.exp(np.quantile(mf[:, 0], 0.025))
        
        aiEst[:, t]   = np.exp(np.mean(mai, axis= 0))
        aiUpper[:, t] = np.exp(np.quantile(mai, 0.975, axis = 0))
        aiLower[:, t] = np.exp(np.quantile(mai, 0.025, axis = 0))
        
        bjEst[:, t]   = np.exp(np.mean(mbj, axis= 0))
        bjUpper[:, t] = np.exp(np.quantile(mbj, 0.975, axis = 0))
        bjLower[:, t] = np.exp(np.quantile(mbj, 0.025, axis = 0))
        
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
        gijEst[:, t]   = np.exp(np.mean(mgij, axis= 0))
        gijUpper[:, t] = np.exp(np.quantile(mgij, 0.975, axis = 0))
        gijLower[:, t] = np.exp(np.quantile(mgij, 0.025, axis = 0))
            
        # gijEst_exp[:, t] = np.mean(np.exp(mgij), axis= 0)
        # gijUpper_exp[:, t] = np.quantile(np.exp(mgij), 0.975, axis = 0)
        # gijLower_exp[:, t] = np.quantile(np.exp(mgij), 0.025, axis = 0)
        
        tmp_idx = np.random.randint(0, sampleSize, size=1)[0]
        MC_f[0, t] = np.exp(mf[tmp_idx, 0])
        MC_ai[:, t] = np.exp(mai[tmp_idx, :])
        MC_bj[:, t] = np.exp(mbj[tmp_idx, :])
        MC_gij[:, t] = np.exp(mgij[tmp_idx, :])
        
    # Values in original scale
    # fEst_exp = np.zeros((1,TActual));
    # aiEst_exp = np.zeros((I,TActual));
    # bjEst_exp = np.zeros((I,TActual));
    # gijEst_exp = np.zeros((N,TActual));
    # for t in range(TActual):
    #     fEst_exp[0, t] = np.mean(np.exp(mf[:, 0]))
    #     aiEst_exp[:, t] = np.mean(np.exp(mai), axis= 0)
    #     bjEst_exp[:, t] = np.mean(np.exp(mbj), axis= 0)
    #     gijEst_exp[:, t] = np.mean(np.exp(mgij), axis= 0)
        
    
    
    return fEst, fUpper, fLower, \
            aiEst, aiUpper, aiLower, \
            bjEst, bjUpper, bjLower, \
            gijEst, gijUpper, gijLower,\
            MC_f, MC_ai, MC_bj, MC_gij
    # return fEst_exp, fUpper_exp, fLower_exp, aiEst_exp, aiUpper_exp, aiLower_exp, \
    #         bjEst_exp, bjUpper_exp, bjLower_exp, gijEst_exp, gijUpper_exp, gijLower_exp, \
    #         MC_f, MC_ai, MC_bj, MC_gij, \
    #         all_bsSample, all_mf, all_mai, all_mbj, all_mgij
