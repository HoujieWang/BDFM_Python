import os
import scipy.linalg
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from scipy.sparse import lil_array
import time
import datetime
import statsmodels.api as sm
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
import pickle
import datetime
from pyspark import SparkContext, SparkConf
import sys
############################## Prepare the Data ###############################
os.chdir("/Users/wanghoujie/Downloads/DynamicNetwork/all_agents")

all_agent_df = [pd.read_parquet('all_baseline_train_downsampled_with_subpop/' + file) \
               for file in [f for f in os.listdir('all_baseline_train_downsampled_with_subpop') \
                            if not f.startswith('.')]]
agent_location = np.array([np.array(x.space_bin) for x in all_agent_df])
unique_zones = np.unique(agent_location)

agent_cluster = np.array([x.cluster_id[0] for x in all_agent_df])

Nzones = len(unique_zones)
Nagents = len(all_agent_df)
Ntime = agent_location.shape[1]
Ncluster = len(np.unique(agent_cluster))

# First column is original zone id, second column is new zone id
id_map = -np.ones((np.max(unique_zones)+1, 2)).astype("int")
id_map[:,0] = np.arange(0, np.max(unique_zones)+1)

id_map[unique_zones,1] = np.argsort(unique_zones)

# Prepare the agent location in new zone index
agent_location2  = id_map[agent_location.flatten(), 1].reshape(Nagents, Ntime)

# Prepare the center of each zone
temp_mat = np.unique(np.vstack([x[["lon", "lat", "space_bin"]] for x in all_agent_df]), axis=0)
temp_mat = np.hstack([temp_mat, id_map[temp_mat[:,2].astype("int"), 1:2]])

zone_centers = np.zeros((Nzones, 2))
for i in range(Nzones):
    # i = 0
    zone_centers[i,:] = np.mean(temp_mat[temp_mat[:,3] == i, :2], axis=0)

# Prepare occupancy counts
occupancy_counts = np.zeros((Nzones, Ntime))
for t in range(Ntime):
    # t = 0
    occupancy_t = np.unique(agent_location2[:,t],return_counts = True)
    occupancy_counts[occupancy_t[0], t] = occupancy_t[1]



def generateMatricesSparse(listOfSequences, numCells, occupancy, proportion = True):
    """
    Generates transition matrices for time series data.
 
    Parameters
    ----------
    listOfSequences : list
        List of sequences of visited cells for each agent
    numCells : int
        Number of cells in AOI
    occupancy : an numCells-by-numTime array of occupancy counts
        
    Returns
    -------
    List
        A list of 672 sparse matrices
   
    """
    # listOfSequences = agent_location2; numCells = Nzones;
    # occupancy = occupancy_counts; proportion = False
    numTimestamps = len(listOfSequences[0])
    num_Agents = len(listOfSequences)
    matrixList = [None] * numTimestamps
    for t in range(numTimestamps - 1):
        # t = 0
        # print(t)
        mat = lil_array((numCells, numCells))
        # mat = np.zeros((numCells, numCells))
        for a in range(num_Agents):
            # print(a)
            i = listOfSequences[a][t]
            j = listOfSequences[a][t + 1]
            if proportion:
                mat[i, j] += 1 / occupancy[i,t]
            else:
                mat[i, j] += 1 
        # if proportion:
        #     mat[mat != 0] / occupancy[np.where(mat != 0)[1],t]
        # mat = lil_array(mat)
        matrixList[t] = mat.tocsr()
    return matrixList

time_scale = np.array([1, 2, 4, 8, 16])
clustered_trans_mat = [None] * Ncluster
for i in range(Ncluster):
    clustered_trans_mat_i = [None] * len(time_scale)
    for s in range(len(time_scale)):
        print((i, s))
        clustered_trans_mat_i[s] = \
            generateMatricesSparse(agent_location2[np.where(agent_cluster == i)[0]][:, \
                                                   np.arange(0, Ntime, time_scale[s])], Nzones, \
                                   occupancy_counts[:, np.arange(0, Ntime, time_scale[s])], True)
    

    clustered_trans_mat[i] = clustered_trans_mat_i



nSeries = 5
for i in range(Nagent):
    # i = 1001
    agent_i = all_agent_df[i]
    
    for s in range(len(time_scale)):
        # s = 1
        scale = time_scale[s]
        
        y_i = agent_location2[i, :][np.arange(0, Ntime, step = scale)]
        flow_i = np.vstack((y_i[: -1], y_i[1:])).T
        
        
        
        trn_flow_i, trn_node_order = flow_converger(y_i.reshape(len(y_i), 1), max(30, nSeries))
        
        
        occupancy_scale = occupancy_counts[:, np.arange(0, Ntime, step = scale)]


        occu_ratio = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
        trans_dist = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
        emp_trans_prob = -np.ones((flow_i.shape[0], trn_node_order.shape[1]-1))*10
        
       
        
        
        
        for t in range(len(y_i)-1):
            # t = 0
            end_zones = trn_node_order[trn_node_order[:,0] == y_i[t], 1:][0,:]
            end_zones = end_zones[end_zones != -1]
            
            
            
            x_t = occupancy_scale[end_zones, t+1] / occupancy_scale[y_i[t], t]
            
            # dest_occu[t, range(len(x_t))] = occupancy_scale[end_zones, t+1]
            occu_ratio[t, range(len(x_t))] = x_t
            
            
            trans_dist[t, range(len(x_t))] = np.sqrt(np.sum((zone_centers[y_i[t], :] - \
                                                             zone_centers[end_zones, :])**2, axis=1))
                
            emp_trans_prob[t, range(len(x_t))] = np.clip(logit(clustered_trans_mat[agent_cluster[i]][s][t][y_i[t]:y_i[t]+1, end_zones].toarray()), -10, 10)

        trans_probRA1,trans_probFF1, all_piRA1, beta_parm1 = \
            spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = int(96/scale),\
                        X = [], F = np.array([1,1,0]), \
                        delta = np.array([0.98]), pr_var = 0.1)
                 
        trans_probRA2,trans_probFF2, all_piRA2, beta_parm1 = \
            spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = int(96/scale),\
                        X = [occu_ratio, 100*trans_dist, emp_trans_prob], F = np.array([1,1,0]), \
                        delta = np.array([0.98, 0.98, 0.98, 0.98]), pr_var = 0.1)
                 
        plt.plot(trans_probRA2)
        
        plt.plot(log(trans_probRA1 / trans_probRA2))
        
        np.sum(log(trans_probRA1 / trans_probRA2))
        