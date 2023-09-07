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

############################## Prepare the Data ###############################
occupancy_counts = scipy.sparse.load_npz('occupancy_counts.npz')
time_bins = pd.read_csv('time_bins.csv')
space_bins = pd.read_csv('space_bins.csv')
## read in one agent dataframe
agent_df = pd.read_parquet('train/agent=1132.parquet')
all_agent_df = [pd.read_parquet('train/' + file) \
               for file in [f for f in os.listdir('train') if not f.startswith('.')]]
agent_location = np.array([np.array(x.space_bin[np.arange(0, agent_df.shape[0], \
                                                          np.sum(agent_df.time_bin == 0))])\
                           for x in all_agent_df])
agent_location = np.hstack([agent_location, agent_location[:, -2:-1]])
    
Nagents = len(all_agent_df)
time_intl = int(np.sum(agent_df.time_bin == 0) / 60)
period = int(1440/time_intl)
discount_bern = 1
ed_time = np.max(agent_df.time_bin) + 1
pr_prob = 0.8

# agent_location_i = agent_df.space_bin[np.arange(0, agent_df.shape[0], np.sum(agent_df.time_bin == 0))]
# agent_location_i = np.array(agent_location_i)
# agent_location_i = np.concatenate((agent_location_i, np.array([agent_location_i[-1]])))
# agent_location_i = agent_location_i.reshape(len(agent_location_i), 1)


############################## Run the Joint Model ############################
moving_time = np.where(agent_location[:, :-1] != agent_location[:, 1:])
n_move = np.zeros((ed_time, ))
temp = np.unique(moving_time[1], return_counts=True)
n_move[temp[0]] = temp[1]
n_move = n_move.reshape(1, len(n_move))


discount_binom = 0.99
F_binom = np.array([1,1,0])
F_binom = np.repeat(F_binom, n_move.shape[1]).reshape(len(F_binom), n_move.shape[1])

G_binom = scipy.linalg.block_diag(
                                [1],
                                [[np.cos(2*np.pi/period), np.sin(2*np.pi/period)],
                                [-np.sin(2*np.pi/period), np.cos(2*np.pi/period)]]
                                )
delta_binom = scipy.linalg.block_diag(np.matlib.repmat((1-discount_binom)/discount_binom,1,1),\
                                      np.matlib.repmat((1-discount_binom)/discount_binom,2,2))
mt, Ct, at, Rt, rt, st, skipped = \
    FF_Bernoulli2(F_binom, G_binom, delta_binom,  n_move, pr_prob, \
                  nt = np.repeat(Nagents, n_move.shape[1]))

RA_summary = Retro_sampling(at.shape[1], 1, F_binom, G_binom, \
                mt, Ct, at, Rt, \
                nSample = 500, family = "bernoulli")
phi_t = RA_summary[0][0,:]
phi_t = phi_t.reshape(1, len(phi_t))

############################## Run the Individual Model ############################

# for i in range(agent_location.shape[0]):
#     # i = 51
#     agent_location_i = agent_location[i:(i+1), :].T
#     trn_flow_i, trn_node_order  = flow_converger(agent_location_i, 14)
#     trn_nodes = np.unique(agent_location_i[:ed_time+1, :])
    
#     log_occu = log(np.diag(occupancy_counts.toarray()[agent_location_i[:-1,0],:]).reshape(1, ed_time))
#     X = [log_occu, phi_t]
#     X.append(F_bern)
#     F_bern = np.array([1,1,0])
#     F_bern = np.repeat(F_bern, trn_flow_i.shape[1]).reshape(len(F_bern), trn_flow_i.shape[1])
#     # F_bern = np.vstack((phi_t, X, F_bern))
#     F_bern = np.vstack(X)
    
#     G_bern = scipy.linalg.block_diag(
#                                     [[1]],\
#                                     [[1]],\
#                                     [[1]],\
#                                     [[np.cos(2*np.pi/period), np.sin(2*np.pi/period)],
#                                     [-np.sin(2*np.pi/period), np.cos(2*np.pi/period)]]
#                                     )
#     delta_bern = scipy.linalg.block_diag(
#                                         np.matlib.repmat((1-discount_bern)/discount_bern,1,1),\
#                                         np.matlib.repmat((1-discount_bern)/discount_bern,1,1),\
#                                         np.matlib.repmat((1-discount_bern)/discount_bern,1,1),\
#                                         np.matlib.repmat((1-discount_bern)/discount_bern,2,2))
    
    
#     nflow = trn_flow_i.shape[0]
#     final_mt = np.zeros((G_bern.shape[0], nflow))
#     final_Ct = np.zeros((nflow, G_bern.shape[0], G_bern.shape[0]))
#     # lambda_jointRA = np.zeros((nflow, 2, \
#     #                           agent_location_i.shape[0]-ed_time-1))
#     for ii in range(nflow):
#         # ii = 0
#         mt, Ct, at, Rt, rt, st, skipped = \
#             FF_Bernoulli2(F_bern, G_bern, delta_bern,  trn_flow_i[[ii],:], pr_prob)
            
#         final_mt[:, ii] = mt[:, ed_time-1]
#         final_Ct[ii, :, :] = Ct[:, :, ed_time-1]
        
#     # RA_summary = Retro_sampling(at.shape[1], 1, F_bern, G_bern, \
#     #                 mt, Ct, at, Rt, \
#     #                 nSample = 500, family = "none")
#     # lambda_jointRA[i, :, :] = RA_summary[0][:, -ed_time:]
    

def spatialDBCM(y, X=[], F = np.array([1,1,0]), nPeriod=96, nSeries = 14,\
                G_list=[np.array([[1]]), \
                        np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
                                  [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])], \
                delta = np.repeat(1, len(G_list)+len(X)),\
                delta5 = np.arange(0.9, 1.01, 0.02), ed_time=[]):
    # y: a T by 1 matrix with each entry representing the zone index of the agent
    # X: a list of additional covariates, each covariate should be 1 by T
    # F: a vector of baseline predictors, default choice is intercept + seasonal component
    # G_list: list of G matrices corresponding to the baseline predictors
    # delta: a vector of discount factor for each component, 
    # this has to be satisfied:len(delta)=len(X) + len(F)
        
    # y = np.concatenate([agent_location_i[0:1,0:1], \
    #     np.vstack([agent_location_i[1:97,:] for i in range(7)])])
    # ed_time=[]; nSeries = 10
    # X = []; delta5 = np.array([1, 1, 1, 1])
    # F = np.array([1,0]); 
    # G_list=[np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
    #                   [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])]
    
    if len(ed_time) == 0:
        ed_time =y.shape[0]-1
        
    G_list_Xpart = [np.array([[1]]) for i in range(len(X))]
    G_list_Xpart.extend(G_list)
    G_bern = scipy.linalg.block_diag(*G_list_Xpart)
    
    agent_location_i = y
    trn_flow_i, trn_node_order  = flow_converger(agent_location_i, nSeries, reverse=False)
    trn_nodes = np.unique(agent_location_i[:ed_time+1, :])
    F_bern = np.repeat(F, trn_flow_i.shape[1]).reshape(len(F), trn_flow_i.shape[1])
    
    X.append(F_bern)
    F_bern = np.vstack(X)
    
    delta_bern = scipy.linalg.block_diag(*[np.matlib.repmat((1-delta[i])/delta[i],\
                                                            G_list_Xpart[i].shape[0],G_list_Xpart[i].shape[0]) \
                                            for i in range(len(G_list_Xpart))])
     
    nflow = trn_flow_i.shape[0]
    # delta_ii = np.zeros((nflow, ))
    # delta_ii[:len(delta5)] = delta5
    # delta_ii[len(delta5):] = delta5[-1]
    final_mt = np.zeros((G_bern.shape[0], nflow))
    final_Ct = np.zeros((nflow, G_bern.shape[0], G_bern.shape[0]))
    # lambda_jointRA = np.zeros((nflow, 2, \
    #                           agent_location_i.shape[0]-ed_time-1))
    retro_moments = [None]*nflow
    forwd_parm = [None]*nflow
    for ii in range(nflow):
        # ii = 1
        # delta_bern = scipy.linalg.block_diag(*[np.matlib.repmat((1-delta[i]*delta_ii[ii])/(delta[i]*delta_ii[ii]),\
        #                                                         G_list_Xpart[i].shape[0],G_list_Xpart[i].shape[0]) \
        #                                        for i in range(len(G_list_Xpart))])
        # temp = trn_flow_i[[ii],:]
        # temp[0,15:30] = 1
        mt, Ct, at, Rt, rt, st, skipped = \
            FF_Bernoulli2(F_bern, G_bern, delta_bern,  trn_flow_i[[ii],:], pr_prob)
        
        
        # raw_pi = rt/(rt+st)
        # plt.plot(raw_pi.T)
        # raw_pi[:,np.where(trn_flow_i[[ii],:] == 0)[1]]
        forwd_parm[ii] = np.vstack([rt,st])
        
        # final_mt[:, ii] = mt[:, ed_time-1]
        # final_Ct[ii, :, :] = Ct[:, :, ed_time-1]
        
        RA_summary = Retro_sampling(at.shape[1], F_bern, G_bern, \
                        mt, Ct, at, Rt, \
                        nSample = 500, family = "none")
        retro_moments[ii] = RA_summary[0][:, :ed_time]
        # lambda_jointRA[i, :, :] = RA_summary[0][:, -ed_time:]
        
    
    trans_probRA = np.zeros((retro_moments[0].shape[1], ))
    trans_probFF = np.zeros((retro_moments[0].shape[1], ))
    all_piRA = np.zeros((retro_moments[0].shape[1], nflow))
    all_piFF = np.zeros((retro_moments[0].shape[1], nflow))
    for t in range(retro_moments[0].shape[1]):
    # for t in range(30):
        # t = 31
        last_location = agent_location_i[t,0]
        
        
        all_f  = np.array([retro_moments[ii][0,t] for ii in range(nflow)])
        all_q  = np.array([retro_moments[ii][1,t] for ii in range(nflow)])
        
        
        
        beta_parmRA = np.array([least_squares(bern_eq, np.array([1, 1]), \
                              args = (all_f[i], all_q[i]), \
                              bounds = ((0, 0), (float("inf"),float("inf")))).x \
                              for i in range(nflow)])
        piRA = beta_parmRA[:,0] / np.sum(beta_parmRA, axis=1)
        all_piRA[t, :] = piRA
        trn_next_zones = trn_node_order[trn_node_order[:,0] == last_location, 1:][0,:].astype(int)
    
        transition_probRA = np.array([1 - piRA[0]] + \
                                   [np.prod(piRA[:i])*(1-piRA[i]) \
                                    for i in np.arange(1, nflow)] + \
                                       [np.prod(piRA)])
            
        piFF = np.array([forwd_parm[ii][:,t] for ii in range(nflow)])
        piFF = piFF[:,0] / np.sum(piFF, axis=1)
        all_piFF[t, :] = piFF
        transition_probFF = np.array([1 - piFF[0]] + \
                                   [np.prod(piFF[:i])*(1-piFF[i]) \
                                    for i in np.arange(1, nflow)] + \
                                       [np.prod(piFF)])
        
        trans_probRA[t] = transition_probRA[np.where(agent_location_i[t+1,0] == trn_next_zones)[0][0]]
        trans_probFF[t] = transition_probFF[np.where(agent_location_i[t+1,0] == trn_next_zones)[0][0]]
        

    return trans_probRA, trans_probFF, all_piRA, all_piFF
        
nn = 10
all_trans_probRA1 = np.zeros((nn, ed_time))
all_trans_probRA0 = np.zeros((nn, ed_time))
all_trans_probFF1 = np.zeros((nn, ed_time))
all_trans_probFF0 = np.zeros((nn, ed_time))

for i in range(nn):
    print(i)
    # i = 51
    agent_location_i = agent_location[i:(i+1), :].T
    
    
    # y = np.concatenate([np.vstack([agent_location_i[:96,:] for i in range(7)]),\
    #                     agent_location_i[0:1,0:1]])
    y = agent_location_i
    
    
    trans_probRA1,trans_probFF1, all_piRA1, all_piFF1 = spatialDBCM(y = y, nSeries = 10, nPeriod = 96,\
                              X = [], F = np.array([1,1,0]),\
                              G_list=[np.array([[1]]), \
                                      np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
                                                [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])], \
                              delta = np.array([1, 1]))
        
    trans_probRA0,trans_probFF0, all_piRA0, all_piFF0 = spatialDBCM(y = y, nSeries = 10, nPeriod = 96,\
                              X = [], F = np.array([1,1,0]),\
                              G_list=[np.array([[1]]), \
                                      np.array([[np.cos(2*np.pi/nPeriod), np.sin(2*np.pi/nPeriod)],\
                                                [-np.sin(2*np.pi/nPeriod), np.cos(2*np.pi/nPeriod)]])], \
                              delta = np.array([1, 0.8]))
    

    
    all_trans_probRA1[i, :] = trans_probRA1
    all_trans_probRA0[i, :] = trans_probRA0
    all_trans_probFF1[i, :] = trans_probFF1
    all_trans_probFF0[i, :] = trans_probFF0
    
    

lBF = log(trans_probFF1 / trans_probFF0)
plt.plot(lcumBF(lBF))
plt.title("Cumulative Log Bayes Factor w/o log occupancy")

# plt.plot(log(np.mean(all_trans_probRA1[:7,:], axis=0) / np.mean(all_trans_probRA0[:7,:], axis=0)).T)
# plt.plot(lcumBF(log(np.mean(all_trans_probRA1[:7,:], axis=0) / np.mean(all_trans_probRA0[:7,:], axis=0))))
plt.plot(log(trans_probFF1 / trans_probFF0).T)
plt.title("Log Bayes Factor w/o log occupancy")



np.where(trans_prob1 / trans_prob0 < 1)[0]

np.mean(np.isin(np.where(agent_location_i[1:, ] != agent_location_i[:-1, ])[0], \
        np.where(trans_prob1 / trans_prob0 < 1)[0]))
# plt.plot(mt.T)
# plt.legend(['phi_t', 'log(occupancy)', 'intercept', 'seasonal1','seasonal2'], \
#            bbox_to_anchor=(1.05, 1.0), loc='upper left')
# plt.title("State Vector Plot of Agent " + str(i+1))
    
plt.plot(all_piRA[:, :5])
plt.legend(range(all_piRA[:, :5].shape[1]), \
            bbox_to_anchor=(1.05, 1.0), loc='upper left')

# plt.plot(n_move.T / len(agent_location))
# plt.plot(np.mean(RA_summary[1], axis=0).T)
# plt.title("Proportion of Agents in Motion")
# plt.legend(['Empirical', 'Retro'], \
#               bbox_to_anchor=(1.05, 1.0), loc='upper left') 

