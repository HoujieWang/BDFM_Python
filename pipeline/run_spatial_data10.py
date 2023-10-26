import os
import scipy.linalg
import geopandas as gpd
import numpy as np
import pandas as pd
import ast
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
############################## Prepare the Data ###############################
occupancy_counts = scipy.sparse.load_npz('occupancy_counts.npz')
occupancy_counts = occupancy_counts.toarray()
Nzones = occupancy_counts.shape[0]

# We generate a fake zone to denote the s2 cells outside of Knoxville
occupancy_counts = np.vstack((occupancy_counts, np.zeros((1, occupancy_counts.shape[1]))))

time_bins = pd.read_csv('time_bins.csv')
space_bins = pd.read_csv('space_bins.csv')
s2_nei = pd.read_csv("/Users/wanghoujie/Downloads/s2_neighbors_duke.csv")

s2_nei['neighbors'] = s2_nei['neighbors'].apply(ast.literal_eval)
s2_nei['s2_cell_center'] = s2_nei['s2_cell_center'].apply(ast.literal_eval)

nei_zones = pd.DataFrame(s2_nei['neighbors'].to_list(), \
                        columns=['zone_1', 'zone_2', 'zone_3', 'zone_4']).to_numpy()
    
zone_loc = pd.DataFrame(s2_nei['s2_cell_center'].to_list(), \
                        columns=['lat', 'long']).to_numpy()


for i in range(space_bins.shape[0]):
    # i = 1
    zone_i = space_bins.iloc[i,0]
    nei_zones[nei_zones == zone_i] = i

## read in one agent dataframe

all_agent_df = [pd.read_parquet('train/' + file) \
               for file in [f for f in os.listdir('train') if not f.startswith('.')]]
    
agent_df = pd.read_parquet('train/agent=1132.parquet')

agent_location = np.array([np.array(x.space_bin[np.arange(0, x.shape[0], \
                                                          np.sum(x.time_bin == 0))])\
                           for x in all_agent_df])

nei_zones[nei_zones > Nzones] = Nzones
nei_zones = np.vstack((nei_zones, np.repeat(Nzones, nei_zones.shape[1]))).astype(int)

    
Nagents = len(all_agent_df)
time_intl = int(np.sum(agent_df.time_bin == 0) / 60)
period = int(1440/time_intl)
discount_bern = 1
Ntime = np.max(agent_df.time_bin) + 1
ed_time = np.max(agent_df.time_bin) + 1
pr_prob = 0.8

unique_zones = np.unique(agent_location)


plt.plot(zone_loc[:,1], zone_loc[:,0], "p", markersize = 1)
plt.plot(zone_loc[unique_zones,1], zone_loc[unique_zones,0], "p", \
         markersize = 1, color = "red")
plt.plot([-83.925011], [35.955013], "p", markersize = 6,\
         color = "black")
plt.ylabel("longitutude")
plt.xlabel("latitude")
plt.legend(['all zones', 'visited zones', 'Univ of Tenn'])
plt.title("Zone Visualization of Knoxville Data")
plt.show()

######################### Return of Occupancy Counts ##########################
scale = 8
model_parm = np.zeros((agent_location.shape[0], 2))
all_y1 = np.zeros((agent_location.shape[0], int(period/scale)))
full_y1 = np.zeros((7, int(period/scale), agent_location.shape[0]))
full_y2 = np.zeros((7, int(period/scale), agent_location.shape[0]))
for i in range(agent_location.shape[0]):
    y_i = agent_location[i, :][np.arange(0, Ntime, step = scale)]
    
    count_i = occupancy_counts[y_i, np.arange(0, Ntime, step = scale)]
    count_i2 = copy.deepcopy(count_i)
    for j in range(nei_zones.shape[1]):
        # j = 0
        count_i2 += occupancy_counts[nei_zones[y_i, j], np.arange(0, Ntime, step = scale)]
    
    count_i = occupancy_counts[y_i, np.arange(0, Ntime, step = scale)]
    occu_ratio = np.concatenate([np.log(count_i[1:] / count_i[:-1]), np.array([0])])

    occu_ratio2 = np.concatenate([np.log(count_i2[1:] / count_i2[:-1]), np.array([0])])
    
    full_y1[:,:, i] = occu_ratio.reshape(7, int(period/scale))
    full_y2[:,:, i] = occu_ratio2.reshape(7, int(period/scale))
    
    y1 = np.mean(full_y1[:,:, i], 0)
    y2 = np.mean(full_y2[:,:, i], 0)
    
    occu_ratio.reshape(7, int(period/scale))
    
    
    x = np.arange(0, int(period/scale))
    
    model2 = sm.OLS(y2, x).fit()
    
    
    model1 = sm.OLS(y1, x).fit()
    model_parm[i, :] = np.array([model1.params[0], model2.params[0]])
    
    all_y1[i, :] = y1

plt.plot(np.mean(full_y1, 2).T)


start_time = datetime.datetime(2021, 1, 1, 0, 0)
end_time = datetime.datetime(2021, 1, 2, 0, 0)
interval = datetime.timedelta(minutes=15)
time_labels = np.array([(start_time + i*interval).strftime('%H:%M') \
                        for i in range((end_time - start_time) // interval)])

x = time_labels[np.arange(0, period, step = scale)]
for i in range(7):
    plt.plot(x, np.mean(full_y2[i,:,:], 1), marker = "o", markersize=3)
    # plt.plot(x, np.mean(full_y2[i,:,:], 1), "p", markersize=3)
# plt.plot(x, np.mean(full_y2, 2).T)
plt.xticks(x[np.arange(0, len(x), 2)])
scale_min = int(scale*time_intl)
plt.title('Averaged relative occupancy by day-of-week (%i mins)' % (int(scale*time_intl)))
# plt.legend(loc='lower left')
plt.show()


##################### Relative Distance/Occupancy Return ######################

scale = 12
all_dist = np.zeros((7, int(period/scale), agent_location.shape[0]))
for i in range(agent_location.shape[0]):
    # i = 2
    y_i = agent_location[i, :][np.arange(0, Ntime, step = scale)]
    flow_i = np.vstack((y_i[: -1], y_i[1:])).T
    
    
    trans_dist = np.sqrt(np.sum((zone_loc[flow_i[:,1], :] - zone_loc[flow_i[:,0], :])**2, 1))
    trans_dist = np.concatenate((trans_dist, np.array([0])))
    # trans_dist += 1e-5
    # trans_dist = log(trans_dist)
    # plt.plot((trans_dist.reshape(7, int(period/scale))).T)
    all_dist[:,:,i] = trans_dist.reshape(7, int(period/scale))
    mean_dist = np.mean(all_dist[:,:,i], 0)
    # all_dist[i, :] = mean_dist
    # all_dist[i*7: (i+1)*7, :] = trans_dist.reshape(7, int(period/scale))
    # plt.plot(log(mean_dist+1))
    
x = time_labels[np.arange(0, period, step = scale)]
plt.boxplot(all_dist.reshape(-1, int(period/scale)))
# plt.ylim(-0.01, 0.4)
plt.xticks(np.arange(1, len(x)+1, 2), x[np.arange(0, len(x), 2)])
plt.title('Averaged Transition Distance by (%i mins)' % (int(scale*time_intl)))
plt.show()


x = time_labels[np.arange(0, period, step = scale)]
plt.plot(x, np.mean(all_dist, 2).T)
plt.xticks(x[np.arange(0, len(x), 1)])
scale_min = int(scale*time_intl)
plt.title('Averaged Daily Transition Distance (%i mins)' % (int(scale*time_intl)))
plt.show()








##################### Comparng Covariates and Null mode ######################
all_ratio2 = []
all_ratio3 = []
# j = 3
nSeries = 5
for scale in np.array([1, 2, 4, 8, 16]):
    # scale = 1
    occupancy_scale = occupancy_counts[:, np.arange(0, Ntime, step = scale)]
    marginal_prob_ratio2 = np.zeros((agent_location.shape[0], occupancy_scale.shape[1]-1))
    marginal_prob_ratio3 = np.zeros((agent_location.shape[0], occupancy_scale.shape[1]-1))
    for i in range(agent_location.shape[0]):
        # i = 2
        print(i)
        y_i = agent_location[i, :][np.arange(0, Ntime, step = scale)]
        flow_i = np.vstack((y_i[: -1], y_i[1:])).T
        
        trn_flow_i, trn_node_order = flow_converger(y_i.reshape(len(y_i), 1), max(30, nSeries))
        
        
        occu_ratio = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
        trans_dist = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
        dest_occu = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
        for t in range(len(y_i)-1):
            # t = 0
            end_zones = trn_node_order[trn_node_order[:,0] == y_i[t], 1:][0,:]
            end_zones = end_zones[end_zones != -1]
            
            x_t = occupancy_scale[end_zones, t+1] / occupancy_scale[y_i[t], t]
            
            dest_occu[t, range(len(x_t))] = occupancy_scale[end_zones, t+1]
            occu_ratio[t, range(len(x_t))] = x_t
            
            dist_temp = np.sum((np.array(s2_nei.s2_cell_center.iloc[y_i[t]]).reshape(1,-1) - \
                np.array(s2_nei.s2_cell_center.iloc[end_zones].tolist()))**2, 1)
            trans_dist[t, range(len(x_t))] = dist_temp
            
        trans_probRA1,trans_probFF1, all_piRA1, beta_parm1 = \
            spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = int(96/scale),\
                        X = [occu_ratio,100*trans_dist], F = np.array([1,1,0]), \
                        delta = np.array([0.98, 0.98, 0.98, 0.98]), pr_var = 0.1)
                
        trans_probRA2,trans_probFF2, all_piRA2, beta_parm2 = \
            spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = 0,\
                        X = [occu_ratio,100*trans_dist], F = np.array([1]), \
                        delta = np.array([0.98, 0.98, 0.98]), pr_var = 0.1)
        
        trans_probRA3,trans_probFF3, all_piRA1, beta_parm1 = \
            spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = int(96/scale),\
                        X = [dest_occu, occu_ratio,100*trans_dist], F = np.array([1,1,0]), \
                        delta = np.array([0.98, 0.98, 0.98, 0.98, 0.98]), pr_var = 0.1)

        
        marginal_prob_ratio2[i, :] = log(trans_probRA1) - log(trans_probRA2)
        marginal_prob_ratio3[i, :] = log(trans_probRA1) - log(trans_probRA3)
        
    all_ratio2.append(marginal_prob_ratio2)
    all_ratio3.append(marginal_prob_ratio3)
    


with open('all_ratio.pkl', 'wb') as f:
    pickle.dump(all_ratio, f)

with open('all_ratio.pkl', 'rb') as f:
    all_ratio = pickle.load(f)
    

cum_lBF2 = np.zeros((5, agent_location.shape[0]))
cum_lBF3 = np.zeros((5, agent_location.shape[0]))
for i in range(cum_lBF2.shape[0]):
    # cum_lBF2[i, :] = np.array([sum((all_ratio2[i][ii,:])) \
    #             for ii in range(agent_location.shape[0])])
    # cum_lBF3[i, :] = np.array([sum((all_ratio3[i][ii,:])) \
    #             for ii in range(agent_location.shape[0])])
        
    cum_lBF3[i, :] = np.sum(all_ratio3[i], axis=1)
    cum_lBF2[i, :] = np.sum(all_ratio2[i], axis=1)
        
        
        
plt.boxplot(cum_lBF2.T, labels=("15min", "30min", "1h", "2h", "4h")) 
plt.axhline(y=-2, color='r', linestyle='--')
# plt.ylim(-50, 25)
plt.title("Distributon 52 agents' Cum-log Bayes factor by time-scale" )
plt.show()

        
plt.boxplot(cum_lBF3.T, labels=("15min", "30min", "1h", "2h", "4h")) 
plt.axhline(y=-2, color='r', linestyle='--')
plt.ylim(-50, 25)
plt.title("Distributon 52 agents' Cum-log Bayes factor by time-scale" )
plt.show()


def cumBF(x):
    # x = marginal_prob_ratio[0,:]
    lbf = x[0]
    lt = 0
    for t in range(len(x)-1):
        lbf *= min(1.0, x[t+1])
        
        if (x[t+1] < 1):
            lt += 1
        else:
            lt = 1
    return lbf, lt

def lcumBF(x):
    # x = marginal_prob_ratio[0,:]
    lbf = x[0]
    lt = 0
    for t in range(len(x)-1):
        lbf  += min(0, x[t+1])
        
        if (x[t+1] < 0):
            lt += 1
        else:
            lt = 1
    return lbf



nSeries = 22
all_scales = np.array([1, 2, 4, 8, 16])
all_pr_var = np.array([0.1, 1])
all_delta = np.array([0.85, 0.95, 1])

index = pd.MultiIndex.from_product([all_scales, all_delta, all_pr_var], \
                                   names=['scale', 'discount', "pr var"])
parm_grid = pd.DataFrame(index=index).reset_index()
parm_grid = np.array(result)

retro_probs = [None]*len(parm_grid)
for i in range(len(parm_grid)):
    # i = 0
    print(i)
    scale = parm_grid[i, 0]
    delta = parm_grid[i, 1]
    pr_var = parm_grid[i, 2]
    
    
    occupancy_scale = occupancy_counts[:, np.arange(0, Ntime, step = int(scale))]
    marginal_probs = np.zeros((4, occupancy_scale.shape[1]-1, agent_location.shape[0]))
    
    
    y_i = agent_location[0, :][np.arange(0, Ntime, step = int(scale))]
    flow_i = np.vstack((y_i[: -1], y_i[1:])).T
    
    trn_flow_i, trn_node_order = flow_converger(y_i.reshape(len(y_i), 1), nSeries)
    
    
    occu_ratio = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
    trans_dist = np.zeros((flow_i.shape[0], trn_node_order.shape[1]-1))
    for t in range(len(y_i)-1):
        # t = 0
        end_zones = trn_node_order[trn_node_order[:,0] == y_i[t], 1:][0,:]
        end_zones = end_zones[end_zones != -1]
        
        x_t = occupancy_scale[end_zones, t+1] / occupancy_scale[y_i[t], t]
        
        occu_ratio[t, range(len(x_t))] = x_t
        
        dist_temp = np.sum((np.array(s2_nei.s2_cell_center.iloc[y_i[t]]).reshape(1,-1) - \
            np.array(s2_nei.s2_cell_center.iloc[end_zones].tolist()))**2, 1)
        trans_dist[t, range(len(x_t))] = dist_temp
        
    trans_probRA1,trans_probFF1, all_piRA1, all_piFF1 = \
        spatialDBCM(y = y_i.reshape(-1, 1), nSeries = nSeries, nPeriod = int(96/scale),\
                    X = [occu_ratio,111*trans_dist], F = np.array([1,1,0]), \
                    delta = np.repeat(delta,4), FF_prob = False, pr_var = pr_var)
        

    retro_probs[i] = trans_probRA1

for i in range(len(all_scales)):
    # i= 2
    scale = all_scales[i]
    parm_i = parm_grid[i*6: (i+1)*6, :]
    plot_data = np.array(retro_probs[i*6: (i+1)*6]).T
    
    plot_data1 = plot_data[:,np.arange(0, 6, 2)]
    plot_data2 = plot_data[:,np.arange(1, 6, 2)]
    
    labels =["delta=%.2f" % (all_delta[ii]) for ii in range(len(all_delta))]
    
    
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    
    for j in range(plot_data1.shape[1]):
        axs[0].plot(np.arange(0,len(plot_data1)), (plot_data1[:,j]), \
                    label = labels[j], linewidth=1)
        # axs[0].plot(np.arange(0,len(plot_data1)), plot_data1[:,j], \
        #             "p", markersize=3, color = "black")
    axs[0].set_title("Marginal transition prob time-scale=1h; init var=%.1f" % (all_pr_var[0]))
    # plt.legend()
    # plt.show()
    for j in range(plot_data1.shape[1]):
        axs[1].plot(np.arange(0,len(plot_data2)), (plot_data2[:,j]), \
                    label = labels[j], linewidth=1)
        # axs[1].plot(np.arange(0,len(plot_data2)), plot_data2[:,j], \
        #             "p", markersize=3, color = "black")
    axs[1].set_title("Marginal transition prob time-scale=1h; init var=%.1f" % (all_pr_var[1]))
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()
  
parm_grid[parm_grid[:,0] == 8,:]
plt.plot(np.array(retro_probs[-2:]).T)

plt.plot(all_probs[0][0,:,0].T, linewidth = 1)
plt.ylim(0, 1)


plt.plot(log((all_probs[3][:3,:,0] / all_probs[3][3,:,0]).T))
plt.ylim(-5, 5)

plt.plot(trans_probRA1/trans_probRA2)
plt.ylim(0, 5)

# marginal_prob_ratio16 = marginal_prob_ratio
# marginal_prob_ratio8 = marginal_prob_ratio


# marginal_prob_ratio = np.array([lcumBF(-log(marginal_prob_ratio[i,:])) \
#            for i in range(agent_location.shape[0])]) 
# plt.hist(marginal_prob_ratio)
# plt.title("Histogram of cumulative log Bayes factor over 52 agents")


# cum_lBF = np.zeros((5, agent_location.shape[0]))
# for i in range(cum_lBF.shape[0]):
#     cum_lBF[i, :] = np.array([lcumBF(-log(all_ratio[i][ii,:])) \
#                 for ii in range(agent_location.shape[0])])

cum_lBF = np.zeros((5, agent_location.shape[0]))
for i in range(cum_lBF.shape[0]):
    cum_lBF[i, :] = np.array([sum(-log(all_ratio[i][ii,:])) \
                for ii in range(agent_location.shape[0])])
        
        
        
plt.boxplot(cum_lBF.T, labels=("15min", "30min", "1h", "2h", "4h")) 
plt.axhline(y=-2, color='r', linestyle='--')
plt.ylim(-50, 25)
plt.title("Distributon 52 agents' Cum-log Bayes factor by time-scale" )
plt.show()








# plt.hist(marginal_prob_ratio1, bins=10, alpha=0.5, label='2h', color='red')
# plt.hist(marginal_prob_ratio, bins=10, alpha=0.5, label='4h', color='blue')

# # Customize plot
# plt.title('Histogram of cum-log Bayes factor over 52 agents')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend(loc='upper left')

# # Show the plot
# plt.show()


# plt.plot(-log((marginal_prob_ratio).T))
# plt.title("Bayes Factor w/o occupancy")
# plt.ylim(-10, 5)


# plt.plot(np.prod(marginal_prob_ratio, 1))
# plt.ylim(-1, 5)
# plt.plot(log(trans_probFF1 / trans_probFF0).T)
# plt.title("Log Bayes Factor w/o log occupancy")




















