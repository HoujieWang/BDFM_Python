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
time_intl = 15
nvertical = 4; nhorizontal = 10
horizontal_grid = np.arange(0, nhorizontal).reshape(nhorizontal, 1)
vertical_grid = np.arange(0, nvertical).reshape(nvertical, 1)
# oneday_obs = int(24 / (time_intl / 60))

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



y15 = agent_location_i
out15 = spatialDBCM(y = y15, ed_time = int((len(agent_location_i)-1)/2), \
                    nSeries = 5, nPeriod = 96,\
                    X = [], F = np.array([1,1,0]), \
                    delta = np.array([1, 1, 1]), dist_mat = dist_mat)

y30 = np.vstack([agent_location_i[0:1,0:1], agent_location_i[np.arange(1, len(agent_location_i), 2), :]])
out30 = spatialDBCM(y = y30, ed_time = int((len(y30)-1)/2), \
                    nSeries = 5, nPeriod = 48,\
                    X = [], F = np.array([1,1,0]), \
                    delta = np.array([1, 1, 1]), dist_mat = dist_mat)

y60 = np.vstack([agent_location_i[0:1,0:1], agent_location_i[np.arange(1, len(agent_location_i), 4), :]])
out60 = spatialDBCM(y = y60, ed_time = int((len(y60)-1)/2), \
                    nSeries = 5, nPeriod = 24,\
                    X = [], F = np.array([1,1,0]), \
                    delta = np.array([1, 1, 1]), dist_mat = dist_mat)

y120 = np.vstack([agent_location_i[0:1,0:1], agent_location_i[np.arange(1, len(agent_location_i), 8), :]])
out120 = spatialDBCM(y = y120, ed_time = int((len(y120)-1)/2), \
                    nSeries = 5, nPeriod = 12,\
                    X = [], F = np.array([1,1,0]), \
                    delta = np.array([1, 1, 1]), dist_mat = dist_mat)

main_prod = np.prod(out15[1].reshape(int(len(out15[1])/2), 2), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out30[1])
plt.title("Marginal Transition Prob 15->30min vs. 30min")


main_prod = np.prod(out15[1].reshape(int(len(out15[1])/4), 4), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out60[1])
plt.title("Marginal Transition Prob 15->60min vs. 60min")


main_prod = np.prod(out15[1].reshape(int(len(out15[1])/8), 8), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out120[1])
plt.title("Marginal Transition Prob 15->120min vs. 120min")


main_prod = np.prod(out30[1].reshape(int(len(out30[1])/2), 2), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out60[1])
plt.title("Marginal Transition Prob 30->60min vs. 60min")


main_prod = np.prod(out30[1].reshape(int(len(out30[1])/4), 4), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out120[1])
plt.title("Marginal Transition Prob 30->120min vs. 120min")

main_prod = np.prod(out60[1].reshape(int(len(out60[1])/2), 2), 1)
main_prod = main_prod + (1 - main_prod)*(1 / nvertical / nhorizontal)
plt.plot(main_prod)
plt.plot(out120[1])
plt.title("Marginal Transition Prob 60->120min vs. 120min")

















