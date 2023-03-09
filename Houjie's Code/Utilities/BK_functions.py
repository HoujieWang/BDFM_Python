import numpy as np

def RA_Poisson_lambda(rt_pois_all, st_pois_all, delta_pois, samps = 1000):

    #ss_theta_pois = np.zeros(rt_pois_all.shape)
    #Analytically for means
    # ss_theta_pois[T - 1, :] = rt_pois_all[T-1, :] * st_pois_all[T-1, :]
    # for t in range(2, T + 1):
    #     ss_theta_pois[T - t, :] = (delta_pois * rt_pois_all[T - t + 1, :] * st_pois_all[T - T + 1, :] + 
    #                            (1 - delta_pois) * rt_pois_all[T - t, :] * st_pois_all[T - t, :])
    T = rt_pois_all.shape[0]
    #Backsampling
    backsample = np.zeros([rt_pois_all.shape[0], rt_pois_all.shape[1], samps])
    for s in np.arange(0, samps):
        backsample[-1, :, s] = np.random.gamma(rt_pois_all[-1, :], st_pois_all[-1, :])
        for t in np.arange(-2, -(T + 1), -1):
            epsilon = np.random.gamma((1 - delta_pois) * rt_pois_all[t, :], st_pois_all[t, :])
            backsample[t, :, s] = delta_pois * backsample[t + 1, :, s] + epsilon
            
    ss_mean = np.mean(backsample, 2)
    ss_lower = np.quantile(backsample, .025, 2)
    ss_upper = np.quantile(backsample, .975, 2)
    
    return(ss_mean, ss_lower, ss_upper)

