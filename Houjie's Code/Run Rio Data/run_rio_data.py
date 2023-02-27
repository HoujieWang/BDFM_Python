import os
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
import pandas as pd
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib.colors as mcolors
from Poisson import FF_Poisson, RA_Poisson
from Bernoulli import FF_Bernoulli, RA_Bernoulli
from Recouple_DGM2 import Recouple_DGM2
#### Load Rio bus data and shapfile 

os.chdir('/Users/wanghoujie/Downloads/DynamicNetwork/bus_gps_data')

busData = pd.read_csv('treatedBusDataOnlyRoute.csv')
sf = gpd.read_file('data/33MUE250GC_SIR.shp')
rio = sf[sf.ID == 1535] # ID of the city of Rio


# Generate spatial grid points
rio_bounds = np.array(rio.bounds).reshape(4,)
cell_size = 0.025 # grid cell size 
lon_coords = np.arange(rio_bounds[0], rio_bounds[2]+cell_size, cell_size) 
lat_coords = np.arange(rio_bounds[1], rio_bounds[3]+cell_size, cell_size)
rio_grid = np.array(np.meshgrid(lon_coords, lat_coords)).\
    reshape(2, len(lon_coords)*len(lat_coords)).T


# Specify a specific bus line within a day
all_bus_line = np.unique(busData.line)
bus_date = '01-25-2019'; 
bus_date = ['01-25-2019', '01-26-2019']
sub_bus = busData[np.isin(np.array(busData.line), all_bus_line[0: 30]) &  \
                  np.isin(busData.date, bus_date) \
                    # & (busData.order == bus_order)
                    ]


bus_time = np.array([np.datetime64(datetime.strptime(t, '%m-%d-%Y%H:%M:%S')) \
                        for t in np.array(sub_bus.date) + np.array(sub_bus.time)])

# Generate time grid points
time_intl = '30min'
time_grid = pd.date_range(bus_date[0], periods=49*len(bus_date), freq=time_intl)
# time_grid = time_grid[(np.where(time_grid >= np.min(bus_time))[0][0]-1): len(time_grid)]

bus_id = np.unique(sub_bus.order)
bus_location = (np.zeros((len(bus_id), len(time_grid)-1))-1).astype("int")
for i in range(len(time_grid)-1):
    # i = 31
    print(i)
    t0 = time_grid[i]
    t1 = time_grid[i+1]
    sub_bus_tmp = sub_bus.iloc[(t0 < bus_time) & (bus_time < t1), ]
    for j in range(len(bus_id)):
        print("Running bus", j, "at time ", t0)
        # bus = 'D13022'
        bus = bus_id[j]
        if (sum(sub_bus_tmp.order == bus) != 0):
            sub_bus_tmp_i = sub_bus_tmp.iloc[np.array(sub_bus_tmp.order == bus), :]
            # the point is the largest lat,long point that are smaller than 
            # the bus's latest lat,long point in the current time interval
            lon_point = lon_coords[np.where(lon_coords > sub_bus_tmp_i.longitude.iloc[-1])[0][0]-1]
            lat_point = lat_coords[np.where(lat_coords > sub_bus_tmp_i.latitude.iloc[-1])[0][0]-1]
            
            index = np.where((rio_grid[:,0] == lon_point) & (rio_grid[:,1] == lat_point))[0][0]
            bus_location[j, i] = index
            
        
# After we obtain the bus location at each period, we get the flows
edge_idx = [None] * (bus_location.shape[1]-1)
last_location = bus_location[:, 0]
for i in np.arange(1,bus_location.shape[1]):
    # i = 1
    bus_location_i = np.vstack([last_location, bus_location[:, i]]).T
    # only check buses with observed location in both periods
    edgeIndex_tmp = bus_location_i[np.sum(bus_location_i[:, 0:2] != -1, axis = 1) == 2, ] 
    last_location[bus_location_i[:, 1] != -1] = bus_location_i[bus_location_i[:, 1] != -1, 1]
    edge_idx[i-1] = edgeIndex_tmp[edgeIndex_tmp[:, 0] != edgeIndex_tmp[:, 1], :]
    
    


unique_edges = np.unique(np.vstack([edge_idx[i] for i in range(len(edge_idx))]), axis = 0)
flow_count = np.zeros((unique_edges.shape[0], len(edge_idx)))

unique_nodes = np.unique(unique_edges)
out_flows = np.zeros((len(unique_nodes), len(edge_idx)))
for j in range(len(edge_idx)):
    # edge_i = np.array([981, 1053])
    for k in range(len(unique_nodes)):
        # k = 0
        node = unique_nodes[k]
        out_flows[k, j] = np.sum(edge_idx[j][:, 0] == node)
        
    for i in range(unique_edges.shape[0]):
        edge_i = unique_edges[i, :]
        flow_count[i, j] = \
        np.sum((edge_i[0] == edge_idx[j][:, 0]) & (edge_i[1] == edge_idx[j][:, 1]))
        
out_flows = out_flows.T      

# Now the flow_count, unique_edges, m are the flows for each edge, edge ID, occupancy ratio
TTotal, I = out_flows.shape
T0= 0
TActual = TTotal - T0
m = np.zeros((TActual,I), dtype = 'double')
m[:, (I-1)] = 1
for t in range(TActual):
    for i in range(I):
        if out_flows[t+T0,i] == 0:
            m[t,i] = 0;
        elif out_flows[t+T0, i] <= 10 or out_flows[t+T0-1, i] <= 10:
            m[t, i] = 1
        else:
            m[t, i] = out_flows[t+T0, i]/out_flows[t+T0-1, i]


############################## Fitting DCMM ##################################


eps = 1e-2
flow_count[flow_count == 0] = eps

N = flow_count.shape[0]

# Bernoulli

F_bern = np.array([[1],
                   [0]])
G_bern = np.array([[1, 1], 
                   [0, 1]])
delta_bern = 0.95  # Discount factor for evolution variance
sampleSize_bern = 1000 # Sample size of posterior and predictions

# Poisson
F_pois = np.array([[1],
                   [0]])
G_pois = np.array([[1, 1], 
                   [0, 1]])
delta_pois = 0.90  # Discount factor for evolution variance
RE_rho_pois = 0.90 # Discount factor for random effect. When = 1, no RE
conditional_shift_pois = 1 # Shift of conditional Poisson
sampleSize_pois = 1000 # Sample size of posterior and predictions

sampleSize_dgm = 500
# Fit all individual dynamic models
phi_samp = np.zeros((N, TActual, sampleSize_dgm))

# Get learned parameters
rt_bern_all = np.zeros((TActual, N))
st_bern_all = np.zeros((TActual, N))
rt_pois_all = np.zeros((TActual, N))
st_pois_all = np.zeros((TActual, N))

ssrt_bern_all = np.zeros((TActual, N))
ssst_bern_all = np.zeros((TActual, N))
ssrt_pois_all = np.zeros((TActual, N))
ssst_pois_all = np.zeros((TActual, N))


# Get sample of phi
for n in range(N):
    # n = 0
    print(n)
    # Bern: Forward Filtering
    mN = m[:,np.where(unique_nodes == unique_edges[n,0])[0]]
    
    st = time.time()
    mt_bern, Ct_bern, at_bern, Rt_bern, rt_bern, st_bern, skipped_bern = \
    FF_Bernoulli(F_bern, G_bern, delta_bern, flow_count, n, T0, TActual, eps)
    ed = time.time()
    # ed - st

    # Bern: Retrospective Analysis
    st = time.time()
    [sat_bern, sRt_bern, ssrt_bern, ssst_bern] = \
        RA_Bernoulli(TActual, F_bern, G_bern, mt_bern, Ct_bern, at_bern, Rt_bern, skipped_bern)
    ed = time.time()
    # ed - st
    
    # Forward Filtering
    st = time.time()
    [mt_pois, Ct_pois, at_pois, Rt_pois, rt_pois, ct_pois, skipped_pois] = \
        FF_Poisson(F_pois, G_pois, delta_pois, flow_count, n, mN, T0, TActual, \
                   eps, RE_rho_pois, conditional_shift_pois)
    ed = time.time()
    # ed - st
    
    # Retrospective Analysis
    [sat_pois, sRt_pois, ssrt_pois, ssct_pois] = \
        RA_Poisson(TActual, F_pois, G_pois, mt_pois, \
                   Ct_pois, at_pois, Rt_pois, skipped_pois)
    
    # Store them
    rt_bern_all[:, n] = rt_bern
    st_bern_all[:, n] = st_bern
    rt_pois_all[:, n] = rt_pois
    st_pois_all[:, n] = ct_pois

    ssrt_bern_all[:, n] = ssrt_bern
    ssst_bern_all[:, n] = ssst_bern
    ssrt_pois_all[:, n] = ssrt_pois
    ssst_pois_all[:, n] = ssct_pois




# Decompose fitted data
fEst, fUpper, fLower, aiEst, aiUpper, aiLower, bjEst, bjUpper, bjLower, \
    gijEst, gijUpper, gijLower = \
    Recouple_DGM2(rt_bern_all, st_bern_all, rt_pois_all, st_pois_all,\
    conditional_shift_pois, \
    TActual, unique_edges, unique_nodes, 1000, I, N)


fEst_r, fUpper_r, fLower_r, aiEst_r, aiUpper_r, aiLower_r, bjEst_r, bjUpper_r, bjLower_r, \
    gijEst_r, gijUpper_r, gijLower_r = \
    Recouple_DGM2(ssrt_bern_all, ssst_bern_all, ssrt_pois_all, ssst_pois_all,\
    conditional_shift_pois, \
    TActual, unique_edges, unique_nodes, 1000, I, N);



############################ Plotting the DGM parameters and map ############################
# plot_time = np.array([x.time().isoformat() for x in time_grid[np.arange(1, TActual+1)]])
time_grid
plot_time = np.array([x.date().isoformat() + "\n" + x.time().isoformat() \
                      for x in time_grid[np.arange(1, TActual+1)]])

base_plot = plt.plot(plot_time, np.exp(fEst[0, :]))
plt.xticks(plot_time[np.arange(0, len(plot_time), 20, dtype=int)])
plt.title("Baseline process")
# plt.savefig('baseline.pdf')
plt.show()

# for i in range(aiEst_r.shape[0]):
temp_idx = np.flip(np.argsort(np.mean(aiEst_r, axis=1))[np.arange(-5, 0)])
# temp_idx = range(0, aiEst_r.shape[0])
alpha_plot = plt.plot(plot_time, np.exp(aiEst_r[temp_idx, :]).T)
plt.xticks(plot_time[np.arange(0, len(plot_time), 20, dtype=int)])
plt.title("outflow process (alpha_i)")
lgd = plt.legend(labels=unique_nodes[temp_idx],
           bbox_to_anchor=(1.04, 1), 
           loc="upper left")
# plt.savefig('outflow.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()
# temp_idx = np.argsort(np.mean(bjEst_r, axis=1))[np.arange(-10, 0)]
# temp_idx = range(0, bjEst_r.shape[0])
beta_plot = plt.plot(plot_time, np.exp(bjEst_r[temp_idx, :]).T)
plt.xticks(plot_time[np.arange(0, len(plot_time), 20, dtype=int)])
plt.title("inflow process (beta_j)")
lgd = plt.legend(labels=unique_nodes[temp_idx],
           bbox_to_anchor=(1.04, 1), 
           loc="upper left")
# plt.savefig('inflow.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()




bus_plot = rio.plot(alpha=0.2)
k = 0
for i in np.unique(sub_bus['line']):
    # i = 6
    bus_tmp = sub_bus.loc[sub_bus['line'] == i, :]
    xy_tmp = np.array(bus_tmp[['longitude', 'latitude']])
    plt.plot(xy_tmp[:,0], xy_tmp[:,1], ".", markersize=0.01,\
             color = list(mcolors.CSS4_COLORS.keys())[k])
    k +=1
    
for i in range(rio_grid.shape[0]):
    plt.vlines(x = rio_grid[i, 0], ymin = np.min(rio_grid[:, 1]), 
                ymax =  np.max(rio_grid[:, 1]),
            colors = 'grey', linewidth = 0.5)
    plt.hlines(y = rio_grid[i, 1], xmin = np.min(rio_grid[:, 0]), 
                xmax =  np.max(rio_grid[:, 0]),
            colors = 'grey', linewidth = 0.5)

for i in unique_nodes:
    plt.text(rio_grid[i, 0]+cell_size*0.1, rio_grid[i, 1]+cell_size*0.1,\
    str(i), size = 5)
    

for i in unique_nodes[temp_idx]:
    plt.vlines(x = rio_grid[i, 0], 
                ymin = rio_grid[i, 1], 
                ymax = rio_grid[i, 1]+cell_size,
            colors = 'red', linewidth = 1)
    plt.vlines(x = rio_grid[i, 0]+cell_size, 
                ymin = rio_grid[i, 1], 
                ymax = rio_grid[(i+1), 1]+cell_size,
            colors = 'red', linewidth = 1)
    
    plt.hlines(y = rio_grid[i, 1], 
                xmin = rio_grid[i, 0], 
                xmax = rio_grid[i, 0]+cell_size,
            colors = 'red', linewidth = 1)
    plt.hlines(y = rio_grid[i, 1]+cell_size, 
                xmin = rio_grid[i, 0], 
                xmax = rio_grid[i, 0]+cell_size,
            colors = 'red', linewidth = 1)

# bus_plot.set_title("Trajectory of four buses on 01/25/2019 (9am-5pm)")
# plt.savefig('Rio_map.pdf')
plt.show()


temp_idx = np.flip(np.argsort(np.mean(gijEst_r, axis=1))[np.arange(-10, 0)])
plt.plot(plot_time, np.exp(gijEst_r[temp_idx, :]).T)
plt.xticks(plot_time[np.arange(0, len(plot_time), 20, dtype=int)])
lgd = plt.legend(labels=unique_edges[temp_idx, :],
           bbox_to_anchor=(1.04, 1), 
           loc="upper left")
plt.title("affinity process (gamma_ij)")
# plt.savefig('affinity.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.show()

