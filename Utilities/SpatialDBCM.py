import os
import scipy.linalg
import geopandas as gpd
import numpy as np
import pandas as pd
import time
from scipy.special import logit, expit, loggamma, polygamma
from scipy.stats import norm, shapiro, beta
from numpy import exp, log, quantile, log10
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
import copy
from datetime import datetime
from Poisson import FF_Poisson2
from Bernoulli import *
from flow_counter import *
import scipy.sparse
from scipy.sparse import csr_matrix
import pyarrow



def spatialDBCM(y, X=[], F = np.array([1, 1, 0]), \
                delta = np.array([1, 1, 1]), \
                nPeriod=96, nSeries = 5, ed_time=[], RA_prob = True, FF_prob = False, pr_var = 0.1):
    # y: a T by 1 matrix with each entry representing the zone index of the agent
    # X: a list of additional covariates, each covariate should be 1 by T
    # F: a vector of baseline predictors, default choice is intercept + seasonal component
    # G_list: list of G matrices corresponding to the baseline predictors
    # delta: a vector of discount factor for each component, this has to be satisfied:len(delta)=len(X) + len(F)
    # RA_prob = True means you want the backward posterior marginal transition probability
    # FF_prob = True means you want the forward marginal transition probability
    # pr_var is the prior variance C0
    # nSeries is the number of top-k zones you would like to fit
        
    # y = y_i.reshape(-1, 1)
    # nSeries = nSeries
    # nPeriod = int(96/scale)
    # X = [occu_ratio, 100*trans_dist, emp_trans_prob]
    # F = np.array([1,1,0])
    # delta = np.array([0.98, 0.98, 0.98, 0.98])
    # ed_time = []
    # pr_var = 0.1
    if len(ed_time) == 0:
        ed_time =y.shape[0]-1
    
    if nPeriod == 0: # no seasonal component
        G_list=[np.array([[1]])]
    else: 
        G_list=[np.array([[1]]), \
                np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
                          [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])]
    
        
    G_list_Xpart = [np.array([[1]]) for i in range(len(X))]
    G_list_Xpart.extend(G_list)
    G_bern = scipy.linalg.block_diag(*G_list_Xpart)
    
    agent_location_i = y
    trn_flow_i, trn_node_order  = flow_converger(agent_location_i, max(30, nSeries), reverse=False)
    trn_nodes = np.unique(agent_location_i[:ed_time+1, :])
    F_bern = np.repeat(F, trn_flow_i.shape[1]).reshape(len(F), trn_flow_i.shape[1])
    
    # X.append(F_bern)
    # F_bern = np.vstack(X)
    
    delta = np.concatenate([delta, np.repeat(1, len(G_list_Xpart)-len(delta))])
    delta_bern = scipy.linalg.block_diag(*[np.matlib.repmat((1-delta[i])/delta[i],\
                                                            G_list_Xpart[i].shape[0],G_list_Xpart[i].shape[0]) \
                                            for i in range(len(G_list_Xpart))])
        
    
     
    # nflow = trn_flow_i.shape[0]
    nflow = nSeries
    final_mt = np.zeros((G_bern.shape[0], nflow))
    final_Ct = np.zeros((nflow, G_bern.shape[0], G_bern.shape[0]))
    retro_moments = [None]*nflow
    forwd_parm = [None]*nflow
    
    
    for ii in range(nflow):
        # ii = 0
        if len(X) > 0:
            F_ii = np.vstack([np.array([X[j][:,ii] for j in range(len(X))]),\
                          F_bern])
        else:
            F_ii = F_bern
                
        mt, Ct, at, Rt, rt, st, skipped = \
            FF_Bernoulli2(F_ii, G_bern, delta_bern,  trn_flow_i[[ii],:], pr_var = pr_var)
        
     
        forwd_parm[ii] = np.vstack([rt,st])
        
               
        RA_summary = Retro_sampling(at.shape[1], F_ii, G_bern, \
                        mt, Ct, at, Rt, \
                        nSample = 500, family = "none")
        retro_moments[ii] = RA_summary[0][:, :ed_time]
        
    
    
    trans_probRA = np.zeros((retro_moments[0].shape[1], ))
    trans_probFF = np.zeros((retro_moments[0].shape[1], ))
    all_piRA = np.zeros((retro_moments[0].shape[1], nflow))
    all_piFF = np.zeros((retro_moments[0].shape[1], nflow))
    all_beta_parm = np.zeros((nflow, 2, retro_moments[0].shape[1]))
    piFF = np.zeros((nflow, ))
    # nflow = 5
    for t in range(retro_moments[0].shape[1]):
        # t = 0
        last_location = agent_location_i[t,0]
        
        trn_next_zones = trn_node_order[trn_node_order[:,0] == last_location, 1:][0,:].astype(int)
        
        
        
        
        
        if RA_prob:
            all_f  = np.array([retro_moments[ii][0,t] for ii in range(nflow)])
            all_q  = np.array([retro_moments[ii][1,t] for ii in range(nflow)])
            beta_parmRA = np.array([least_squares(bern_eq, np.array([1, 1]), \
                                  args = (all_f[i], all_q[i]), \
                                  bounds = ((0, 0), (float("inf"),float("inf")))).x \
                                  for i in range(nflow)])
                
            all_beta_parm[:len(beta_parmRA),:,t] = beta_parmRA
            piRA = beta_parmRA[:,0] / np.sum(beta_parmRA, axis=1)
            all_piRA[t, :len(piRA)] = piRA
            
        
            transition_probRA = np.array([1 - piRA[0]] + \
                                       [np.prod(piRA[:i])*(1-piRA[i]) \
                                        for i in np.arange(1, nflow)] + \
                                           [np.prod(piRA)])
                
            next_zone_order = np.where(agent_location_i[t+1,0] == trn_next_zones)[0][0]
            if next_zone_order+1 > len(beta_parmRA):
                # transition prob. to the other zones
                trans_probRA[t] = transition_probRA[-1]
            else:
                trans_probRA[t] = transition_probRA[next_zone_order]
            
        if FF_prob:
            
            beta_parmFF = np.array([forwd_parm[ii][:,t] for ii in range(nflow)])
            denominator = np.sum(beta_parmFF, axis=1)
            temp_idx = np.where(denominator > 0)[0]
            piFF[temp_idx] = beta_parmFF[temp_idx,0] / denominator[temp_idx]
            
            
            all_piFF[t, :] = piFF
            transition_probFF = np.array([1 - piFF[0]] + \
                                       [np.prod(piFF[:i])*(1-piFF[i]) \
                                        for i in np.arange(1, nflow)] + \
                                           [np.prod(piFF)])
            trans_probFF[t] = transition_probFF[np.where(agent_location_i[t+1,0] == trn_next_zones)[0][0]]
        
        

    return trans_probRA, trans_probFF, all_piRA, all_beta_parm
        

# def spatialDBCM(y, ed_time, X=[], F = np.array([1, 1, 0]), \
#                 delta = np.array([1, 1, 1]), \
#                 nPeriod=96, nSeries = 5, dist_mat = None, pr_prob = 0.5):
#     # y: a T by 1 matrix with each entry representing the zone index of the agent
#     # X: a list of additional covariates, each covariate should be 1 by T
#     # F: a vector of baseline predictors, default choice is intercept + seasonal component
#     # G_list: list of G matrices corresponding to the baseline predictors
#     # delta: a vector of discount factor for each component, 
#     # this has to be satisfied:len(delta)=len(X) + len(F)
        
#     # y = y_i.reshape(-1, 1); ed_time = len(y_i); nSeries = nSeries; nPeriod = 0
#     # X = [occu_ratio,100*trans_dist]; F = np.array([1,1,0]);
#     # delta = np.array([0.98, 0.98, 0.98, 0.98])
#     # dist_mat = None
    
    
                          
    
    # if nPeriod == 0: # no seasonal component
    #     G_list=[np.array([[1]])]
    # else: 
    #     G_list=[np.array([[1]]), \
    #             np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
    #                       [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])]
    
    
#     if dist_mat is None: # if we only care model fit on training set (i.e. no prediction at new zones) 
#         # this setup will order the destination zones simply by visiting frequency  
#         trn_flow_i, trn_node_order  = flow_converger(y, nSeries, \
#                                                      distance = False, dist_mat = dist_mat, \
#                                                          reverse=False)
#     else: # 
#         trn_flow_i, trn_node_order  = flow_converger(y, nSeries, \
#                                                      distance = True, dist_mat = dist_mat, \
#                                                          reverse=False)
            
        
#     G_list_Xpart = [np.array([[1]]) for i in range(len(X))]
#     G_list_Xpart.extend(G_list)
#     G_bern = scipy.linalg.block_diag(*G_list_Xpart)
    
    
    
            
#     tst_node_order = trn_node_order
    
#     trn_nodes = np.unique(y)
#     F_bern = np.repeat(F, trn_flow_i.shape[1]).reshape(len(F), trn_flow_i.shape[1])
    
#     X.append(F_bern)
#     F_bern = np.vstack(X)
    
#     delta_bern = scipy.linalg.block_diag(*[np.matlib.repmat((1-delta[i])/delta[i],\
#                                                             G_list_Xpart[i].shape[0],G_list_Xpart[i].shape[0]) \
#                                             for i in range(len(G_list_Xpart))])
     
#     nflow = trn_flow_i.shape[0]
#     final_mt = np.zeros((G_bern.shape[0], nflow))
#     final_Ct = np.zeros((nflow, G_bern.shape[0], G_bern.shape[0]))
   
    
#     retro_moments = [None]*nflow
#     forwd_parm = [None]*nflow
#     for ii in range(nflow):
#         mt, Ct, at, Rt, rt, st, skipped = \
#             FF_Bernoulli2(F_bern, G_bern, delta_bern,  trn_flow_i[[ii],:])
        
     
#         forwd_parm[ii] = np.vstack([rt,st])
#         final_mt[:, ii] = mt[:, ed_time-1]
#         final_Ct[ii, :, :] = Ct[:, :, ed_time-1]
        
               
#         RA_summary = Retro_sampling(at.shape[1], F_bern, G_bern, \
#                         mt, Ct, at, Rt, \
#                         nSample = 500, family = "none")
#         retro_moments[ii] = RA_summary[0][:, -ed_time:]
        
    
#     all_marginalRA = np.zeros((retro_moments[0].shape[1], ))
#     all_marginalFF = np.zeros((retro_moments[0].shape[1], ))
#     final_mt2 = copy.deepcopy(final_mt)
#     final_Ct2 = copy.deepcopy(final_Ct)
#     final_mt = final_mt2
#     final_Ct = final_Ct2
#     all_transRA = np.zeros((retro_moments[0].shape[1], len(dist_mat)))
#     all_transFF = np.zeros((retro_moments[0].shape[1], len(dist_mat)))
    
#     for t in range(retro_moments[0].shape[1]):
#     # for t in range(5):
#         # t = 5
#         last_location = y[t+ed_time, 0]
        
#         all_M = [(G_bern @ final_Ct[i,:,:] @ G_bern.T) * \
#                   (1 + delta_bern*0) \
#                       for i in range(nflow)]
#         final_Ct = np.array(all_M)
        
#         all_f  = (F_bern[:,(t+ed_time): (t+ed_time+1)].T @ final_mt).reshape(nflow, )
#         all_q  = np.array([(F_bern[:,(t+ed_time): (t+ed_time+1)].T @ \
#                             final_Ct[i,:,:] @ \
#                             F_bern[:,(t+ed_time): (t+ed_time+1)])[0,0] \
#                   for i in range(nflow)])
        
        
        
#         if (np.sum(trn_nodes == last_location) == 0):
#             piFF = np.repeat(pr_prob, nflow)
#             trn_next_zones = np.argsort(dist_mat[last_location,:])[:nflow].astype(int)
#         else:
#             beta_parmFF = np.array([least_squares(bern_eq, np.array([1, 1]), \
#                                   args = (all_f[i], all_q[i]), \
#                                   bounds = ((0, 0), (float("inf"),float("inf")))).x \
#                                   for i in range(nflow)])
#             piFF = beta_parmFF[:,0] / np.sum(beta_parmFF, axis=1)
#             trn_next_zones = trn_node_order[trn_node_order[:,0] == last_location, 1:][0,:].astype(int)
        
#         marginal_probFF = np.array([1 - piFF[0]] + \
#                                    [np.prod(piFF[:i])*(1-piFF[i]) \
#                                     for i in np.arange(1, nflow)] + \
#                                        [np.prod(piFF)])
        
#         trans_probFF = np.zeros((total_zones, )) + marginal_probFF[-1] / (total_zones - nflow)
#         trans_probFF[trn_next_zones] = marginal_probFF[:nflow]

        
#         RA_mu = np.array([retro_moments[ii][0,t] for ii in range(nflow)])
#         RA_var = np.array([retro_moments[ii][1,t] for ii in range(nflow)])
#         beta_parmRA = np.array([least_squares(bern_eq, np.array([1, 1]), \
#                               args = (RA_mu[i], RA_var[i]), \
#                               bounds = ((0, 0), (float("inf"),float("inf")))).x \
#                               for i in range(nflow)])
#         piRA = beta_parmRA[:,0] / np.sum(beta_parmRA, axis=1)
        
#         marginal_probRA = np.array([1 - piRA[0]] + \
#                                    [np.prod(piRA[:i])*(1-piRA[i]) \
#                                     for i in np.arange(1, nflow)] + \
#                                        [np.prod(piRA)])
#         tst_next_zones = trn_next_zones
#         trans_probRA = np.zeros((total_zones, )) + marginal_probRA[-1] / (total_zones - nflow)
#         trans_probRA[tst_next_zones] = marginal_probRA[:nflow]
        
#         tst_next_zones = tst_node_order[tst_node_order[:,0] == last_location, 1:][0,:]
        
#         all_transRA[t, :] = trans_probRA
#         all_transFF[t, :] = trans_probFF
        
#         # all_piFF[t, :] = piFF
#         # all_piRA[t, :] = piRA
        
        
#         all_marginalRA[t] = trans_probRA[y[t+ed_time+1,0]]
#         all_marginalFF[t] = trans_probFF[y[t+ed_time+1,0]]
        
#     out = [all_transRA, all_marginalRA, all_transFF, all_marginalFF]
#     return out