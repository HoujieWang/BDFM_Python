import os
import scipy.linalg
import geopandas as gpd
import numpy as np
import pandas as pd
import time
import pybats
from scipy.special import logit, expit, loggamma, polygamma
from scipy.stats import norm, shapiro, beta
from scipy.optimize import fsolve, least_squares
from pybats.dglm import bern_dglm
from numpy import exp, log, quantile, log10
from pybats.analysis import analysis_dcmm
from pybats.plot import plot_data_forecast
from pybats.point_forecast import median
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks
from pybats.loss_functions import MAPE
import copy
from datetime import datetime
from Poisson import FF_Poisson2
from Bernoulli import FF_Bernoulli2, Retro_sampling2, Retro_sampling
from flow_counter import flow_counter, flow_counter2

st_date = '03-01-2008'
end_date = '04-30-2008'

latitude = []
longitude = []
date = []
time = []

all_file_name = os.listdir()
all_file_name = np.array(all_file_name)[np.argsort([int(''.join(list(x)[:14])) for x in all_file_name])]


for file in all_file_name:
    # file = '20080430013834.plt'
    data_temp = pd.read_fwf(file, skip_rows=range(5), header=None)
    for i in np.arange(6, data_temp.shape[0]):
        # i = 6
        latitude.append(float(data_temp.iloc[i,0].split(',')[0]))
        longitude.append(float(data_temp.iloc[i,0].split(',')[1]))
        date.append(data_temp.iloc[i,0].split(',')[5])
        time.append(data_temp.iloc[i,0].split(',')[6])

data_full = pd.DataFrame({'latitude': latitude,'longitude': longitude,\
                          'date': date,'time': time, \
                          'order': ['167']*len(latitude)})
data_full.date = [datetime.strptime(date_i, '%Y-%m-%d').date().strftime('%m-%d-%Y') \
                  for date_i in data_full.date]

             
data = data_full.iloc[:np.where(data_full.date == end_date)[0][-1]+1, :]


region_bounds = np.array([np.min(data.longitude)*0.99995, np.min(data.latitude)*0.99995, \
                          np.max(data.longitude)*1.00005, np.max(data.latitude)*1.00005])

        
time_intl = 15
agent_location, flow_count, unique_edges, normal_nodes, agent_id, m , dist_mat= \
    flow_counter2(data, agent_id = ['167'], region_bounds=region_bounds, \
                  cell_size=0.01, \
                  st_date = st_date, end_date = end_date,\
                  time_intl = time_intl)

###############################################################################
time_intl = 15
nvertical = 4; nhorizontal = 10
horizontal_grid = np.arange(0, nhorizontal).reshape(nhorizontal, 1)
vertical_grid = np.arange(0, nvertical).reshape(nvertical, 1)
oneday_obs = int(24 / (time_intl / 60))

normal_day = np.array([(1, 1) for i in range(33)] + \
               [(1, 2) for i in range(3)] + \
               [(2, 8)] +\
               [(4, 10) for i in range(12)] + \
               [(3, 10) for i in range(4)] + \
               [(4, 10) for i in range(18)] + \
               [(4, 8)] + \
               [(4, 7) for i in range(2)] + \
               [(4, 4)] + \
               [(3, 2)] + \
               [(1, 1) for i in range(21)])-1
normal_day = normal_day[np.concatenate((np.array([0]),np.arange(int(time_intl / 15), len(normal_day), int(time_intl / 15)))),:]
normal_nodes = np.unique(normal_day, axis=0)

anomaly_day = np.array([(1, 1) for i in range(33)] + \
               [(1, 3) for i in range(3)] + \
               [(2, 8)] +\
               [(4, 10) for i in range(12)] + \
               [(3, 10) for i in range(4)] + \
               [(4, 10) for i in range(18)] + \
               [(4, 8)] + \
               [(4, 6)] + \
               [(4, 4)] + \
               [(3, 2)] + \
               [(1, 1) for i in range(22)])-1
anomaly_day = anomaly_day[np.concatenate((np.array([0]),np.arange(int(time_intl / 15), len(anomaly_day), int(time_intl / 15)))),:]
anomaly_nodes = np.unique(anomaly_day, axis=0)


unique_nondes2D_i = np.unique(np.vstack((normal_nodes, anomaly_nodes)),axis=0)
unique_nodes_i = np.sort(unique_nondes2D_i[:,1]*nvertical + unique_nondes2D_i[:,0])

spatial_grid = np.array([(x, y) for x in range(4) for y in range(10)])
dist_mat = scipy.spatial.distance_matrix(spatial_grid, spatial_grid, 2)
total_zones = len(spatial_grid)



agent_location2D_i = np.vstack([normal_day[0,:], \
                      np.vstack([normal_day[1:,:] for i in range(7)]), \
                      np.vstack([anomaly_day[1:,:] for i in range(7)])])
agent_location_i =  agent_location2D_i[:,1]*nvertical + agent_location2D_i[:,0]
agent_location_i = agent_location_i.reshape(len(agent_location_i), 1)

data = agent_location_i[:ed_time+1, :]



def flow_converger(data, nzones, distance = False, dist_mat=[]):
    unique_nodes_i = np.unique(data)
    flow_i = np.hstack((data[: -1, :], data[1:  , :]))
    unique_edges_i = np.unique(flow_i, axis = 0)
    flow_count_i = np.zeros((unique_edges_i.shape[0], data.shape[0]-1))
    for ii in range(unique_edges_i.shape[0]):
        flow_count_i[ii, np.sum(np.abs(flow_i - unique_edges_i[ii, :]), axis=1) == 0] = 1
    
    node_order = np.zeros((unique_nodes_i.shape[0], nzones+1))-1
    node_order[:,0] = unique_nodes_i
    
    # This code ranks destination zones by their distance
    # for i in range(len(unique_nodes_i)):
    #     # i = 0
    #     node_i = unique_nodes_i[i]
    #     node_choices = np.zeros((node_order.shape[1]-1, ))-1
    #     end_nodes = unique_edges_i[unique_edges_i[:,0] == node_i,1]
    #     end_nodes = end_nodes[np.argsort(-np.sum(data[unique_edges_i[:,0] == node_i, :], axis=1))]
    #     node_choices[:len(end_nodes)] = end_nodes
    #     filling_choices = np.argsort(dist_mat[np.where(unique_nodes_i == node_i)[0][0], :])
    #     filling_choices = filling_choices[(filling_choices != node_i) & (filling_choices != end_nodes)]
    #     node_choices[len(end_nodes):] = filling_choices[1:(node_order.shape[1]-len(end_nodes))]
    #     node_order[i,1:] = node_choices
    
    # This code ranks zones first by frequency of visit
    for i in range(len(unique_nodes_i)):
        # i = 9
        node_i = unique_nodes_i[i]
        node_choices = np.zeros((node_order.shape[1]-1, ))-1
        end_nodes = unique_edges_i[unique_edges_i[:,0] == node_i,1]
        n_visist = np.array([np.sum(flow_i[flow_i[:,0] == node_i, 1] == x) for x in end_nodes])
        nodes_to_order = [end_nodes[n_visist == -x] for x in np.sort(-np.unique(n_visist))]
        if(distance):
            end_nodes = np.concatenate([x[np.argsort(dist_mat[node_i,:][x])]  for x in nodes_to_order])
            node_choices[:len(end_nodes)] = end_nodes
            
            filling_choices = np.argsort(dist_mat[node_i, :])
            filling_choices = filling_choices[np.logical_not(np.isin(filling_choices, np.concatenate((np.array([node_i]), end_nodes))))]
            
            node_choices[len(end_nodes):] = filling_choices[:(node_order.shape[1]-len(end_nodes)-1)]
        else:
            end_nodes = np.concatenate(nodes_to_order)
            node_choices[:len(end_nodes)] = end_nodes
        
        node_order[i,1:] = node_choices
    

    trans_flow_i = np.zeros((node_order.shape[1]-1, len(data)-1))-1
    temp = np.hstack((data[:-1,:], data[1:,:]))
    for t in np.arange(len(data)-1):
        # t = 48
        last_node = data[t,0]
        now_node = data[t+1,0]
        now_node_rank = np.where(node_order[node_order[:,0] == last_node, 1:] == now_node)[1][0]
        binary_y = np.array([1]*now_node_rank + [0])
        trans_flow_i[:len(binary_y), t] = binary_y
    return trans_flow_i, node_order

# trn_flow_i, trn_node_order  = flow_converger(agent_location_i[:ed_time+1, :], dist_mat, 3)
# tst_flow_i, tst_node_order  = flow_converger(agent_location_i[ed_time:, :], dist_mat, 3)
trn_flow_i, trn_node_order  = flow_converger(agent_location_i, dist_mat, 4)
tst_flow_i, tst_node_order  = flow_converger(agent_location_i, dist_mat, 4)
trn_nodes = np.unique(agent_location_i[:ed_time+1, :])
tst_nodes = np.unique(agent_location_i[ed_time:, :])

nflow = trn_flow_i.shape[0]

T0= 0
period = 1440/time_intl
discount_bern = 1
F_bern = np.array([
                    [1],
                    # [1],
                    # [0],
                    [1],
                    [0]
                    ])

G_bern = scipy.linalg.block_diag(
                                [[1]],
                                # [[1, 1], 
                                # [0, 1]],
                                 
                                [[np.cos(2*np.pi/period), np.sin(2*np.pi/period)],
                                [-np.sin(2*np.pi/period), np.cos(2*np.pi/period)]]
                                )
delta_bern = scipy.linalg.block_diag(np.matlib.repmat((1-discount_bern)/discount_bern,1,1),\
                                      np.matlib.repmat((1-discount_bern)/discount_bern,2,2),\
                                      np.matlib.repmat((1-discount_bern)/discount_bern,0,0))
    
ed_time = (len(normal_day)-1)*7
pr_prob = 0.9
final_mt = np.zeros((G_bern.shape[0], nflow))
final_Ct = np.zeros((nflow, G_bern.shape[0], G_bern.shape[0]))
lambda_jointRA = np.zeros((nflow, 2, \
                          agent_location_i.shape[0]-ed_time-1))
# These codes runs the training and testing set together
# for i in range(nflow):
#     # i = 0
#     zt = trans_flow_i[[i],:]
#     mt, Ct, at, Rt, rt, st, skipped = \
#         FF_Bernoulli2(F_bern, G_bern, delta_bern, zt, pr_prob)
#     final_mt[:, i] = mt[:, ed_time-1]
#     final_Ct[i, :, :] = Ct[:, :, ed_time-1]
    
#     RA_summary = Retro_sampling(at.shape[1], 1, F_bern, G_bern, \
#                     mt, Ct, at, Rt, \
#                     nSample = 500, family = "none")
#     lambda_jointRA[i, :, :] = RA_summary[0][:, ed_time:]

# These codes run the training and testing set separately
for i in range(nflow):
    # i = 0
    mt, Ct, at, Rt, rt, st, skipped = \
        FF_Bernoulli2(F_bern, G_bern, delta_bern,  trn_flow_i[[i],:], pr_prob)
        
    final_mt[:, i] = mt[:, ed_time-1]
    final_Ct[i, :, :] = Ct[:, :, ed_time-1]
    
    mt, Ct, at, Rt, rt, st, skipped = \
        FF_Bernoulli2(F_bern, G_bern, delta_bern,  tst_flow_i[[i],:], pr_prob)
        
    RA_summary = Retro_sampling(at.shape[1], 1, F_bern, G_bern, \
                    mt, Ct, at, Rt, \
                    nSample = 500, family = "none")
    lambda_jointRA[i, :, :] = RA_summary[0][:, -ed_time:]
    
    

final_mt2 = copy.deepcopy(final_mt)
final_Ct2 = copy.deepcopy(final_Ct)
final_mt = final_mt2
final_Ct = final_Ct2
pr_prob = 0.9
idx = 0 
BF = np.zeros((agent_location_i.shape[0]-1-ed_time, 1))
trans_prob_FF = np.zeros((agent_location_i.shape[0]-1-ed_time, ))
trans_prob_RA = np.zeros((agent_location_i.shape[0]-1-ed_time, ))
for t in np.arange(ed_time, agent_location_i.shape[0]-1):
# for t in np.arange(ed_time, ed_time+32):
    # t = ed_time+32
    last_location = agent_location_i[t]
    
    final_mt  = G_bern @ final_mt
    # method 0
    all_M = [(G_bern @ final_Ct[i,:,:] @ G_bern.T) * \
              (1 + delta_bern*0) \
                  for i in range(nflow)]
    final_Ct = np.array(all_M)
    all_f  = (F_bern.T @ final_mt).reshape(nflow, )
    all_q  = np.array([(F_bern.T @ final_Ct[i,:,:] @ F_bern)[0,0] \
              for i in range(nflow)])
    
    
    # Check if the last location has been visited in training data at the same time, 
    # if not just use prior
    if ((np.sum(trn_nodes == last_location) == 0) | \
        ( agent_location_i[(idx % oneday_obs)] != agent_location_i[t])):
        all_piFF = np.repeat(pr_prob, nflow)
        
        
        trn_next_zones = np.argsort(dist_mat[last_location,:])[0, :nflow].astype(int)
    else:
        beta_parmFF = np.array([least_squares(bern_eq, np.array([1, 1]), \
                              args = (all_f[i], all_q[i]), \
                              bounds = ((0, 0), (float("inf"),float("inf")))).x \
                              for i in range(nflow)])
        all_piFF = beta_parmFF[:,0] / np.sum(beta_parmFF, axis=1)
        trn_next_zones = trn_node_order[trn_node_order[:,0] == last_location, 1:][0,:].astype(int)
    
    transition_probFF = np.array([1 - all_piFF[0]] + \
                               [np.prod(all_piFF[:i])*(1-all_piFF[i]) \
                                for i in np.arange(1, nflow)] + \
                                   [np.prod(all_piFF)])
    
    full_probFF = np.zeros((total_zones, )) + transition_probFF[-1] / (total_zones - nflow)
    full_probFF[trn_next_zones] = transition_probFF[:nflow]

    
    RA_mu = lambda_jointRA[:, 0, idx]
    RA_var = lambda_jointRA[:, 1, idx]
    beta_parmRA = np.array([least_squares(bern_eq, np.array([1, 1]), \
                          args = (RA_mu[i], RA_var[i]), \
                          bounds = ((0, 0), (float("inf"),float("inf")))).x \
                          for i in range(nflow)])
    all_piRA = beta_parmRA[:,0] / np.sum(beta_parmRA, axis=1)
    
    transition_probRA = np.array([1 - all_piRA[0]] + \
                               [np.prod(all_piRA[:i])*(1-all_piRA[i]) \
                                for i in np.arange(1, nflow)] + \
                                   [np.prod(all_piRA)])
    tst_next_zones = tst_node_order[tst_node_order[:,0] == last_location, 1:][0,:]
    
        
    next_loc_id_tst = np.where(tst_next_zones==agent_location_i[t+1])[0][0]
    
   
    trans_prob_FF[idx] = full_probFF[agent_location_i[t+1]]
    trans_prob_RA[idx] = transition_probRA[next_loc_id_tst]
    # if ( (trans_prob_FF[idx] < 1e-3) & (trans_prob_RA[idx] < 1e-3)):
    #     BF[idx,:] =  (1-trans_prob_FF[idx]) / (1-trans_prob_RA[idx])
    # else:
    #     BF[idx,:] =  trans_prob_FF[idx] / trans_prob_RA[idx]
    BF[idx,:] =  trans_prob_FF[idx] / trans_prob_RA[idx]
    idx += 1

plt.plot(log(BF))

d = 1
xx = log(BF[oneday_obs*d: oneday_obs*(d+1), 0]).reshape(oneday_obs, 1)
plot_time = pd.date_range(start='2023-05-16 00:00:00', end = '2023-05-17 00:00:00', periods=oneday_obs+1)
plot_time = np.array([x.time().isoformat() \
          for x in plot_time[np.arange(0, len(plot_time)-1)]])
plt.plot(plot_time, np.mean(xx[0:oneday_obs, :], axis=1), 'go--', color = "black", markersize = 3)

plt.xticks(plot_time[np.linspace(0, len(plot_time)-1, 5, dtype=int)])
plt.title("Transition-level Log Bayes Factor of the First Day")
plt.ylabel("Log BF")
plt.ylim(-10, 5)
plt.axhline(y = -2, color = 'b')
plt.axvline(x = 33, color = 'b')
plt.axvline(x = 36, color = 'b')
plt.axvline(x = 71, color = 'b')
plt.axvline(x = 75, color = 'b')

# plt.axvline(x = 8, color = 'b')
# plt.axvline(x = 18, color = 'b')
plt.show()
        