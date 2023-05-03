import os
import scipy.linalg
import geopandas as gpd
import numpy as np
import pandas as pd
import time


from Poisson import FF_Poisson2
from Bernoulli import FF_Bernoulli2, Retro_sampling2
from flow_counter import flow_counter

# os.chdir('/Users/wanghoujie/Downloads/DynamicNetwork/bus_gps_data')

busData = pd.read_csv('/Users/wanghoujie/Downloads/DynamicNetwork/bus_gps_data/treatedBusDataOnlyRoute.csv')
busData['longitude'] = pd.to_numeric(busData.longitude, errors='coerce')
busData['latitude'] = pd.to_numeric(busData.latitude, errors='coerce')


sf = gpd.read_file('data/33MUE250GC_SIR.shp')
rio = sf[sf.ID == 1535] # ID of the city of Rio


rio_bounds = np.array(rio.bounds).reshape(4,)

region_bounds  = rio_bounds

cell_size = 0.05 

time_intl = 30

st_date = '02-22-2019'
end_date = '02-24-2019'

bus_subgrid = busData[(-43.3 <= busData.longitude) & (busData.longitude <= -43.2) & \
    (-23 <= busData.latitude) & (busData.latitude <= -22.8)]
busline_subgrid = np.unique(bus_subgrid.line)
bus_date = pd.date_range(st_date, periods=2, freq='24h').strftime('%m-%d-%Y')
sub_bus = busData[np.isin(np.array(busData.line), busline_subgrid[0: 20]) &  \
                  np.isin(busData.date, bus_date) \
                    # & (busData.order == bus_order)
                    ]
loc_history = sub_bus
# Function input, needs to be an array of names of indices
agent_id = np.unique(loc_history.order)
 
agent_location, flow_count, unique_edges, unique_nodes, agent_id, m = \
    flow_counter(loc_history, agent_id, region_bounds, cell_size, st_date, end_date, time_intl)

############################## Fitting DCMM ##################################

TActual = flow_count.shape[1]
T0= 0

I = len(unique_nodes)
eps = 1e-2
flow_count[flow_count == 0] = eps

N = flow_count.shape[0]

# Bernoulli

# F_bern = np.array([[1],
#                     [0]])
# G_bern = np.array([[1, 1], 
#                     [0, 1]])
F_bern = np.array([[1],
                    [0],
                    [1],
                    [0]])

period = 1440/30
G_bern = scipy.linalg.block_diag([[1, 1], 
                                [0, 1]],
                                [[np.cos(2*np.pi/period), np.sin(2*np.pi/period)],
                                [-np.sin(2*np.pi/period), np.cos(2*np.pi/period)]])

delta_bern = 0.95  # Discount factor for evolution variance
# delta_bern = np.diag(np.repeat(delta_bern, len(F_bern)))
# delta_bern = scipy.linalg.block_diag(np.matlib.repmat((1-delta_bern)/delta_bern,2,2))
delta_bern = scipy.linalg.block_diag(np.matlib.repmat((1-delta_bern)/delta_bern,2,2),\
                                      np.matlib.repmat((1-delta_bern)/delta_bern,2,2))
sampleSize_bern = 1000 # Sample size of posterior and predictions

# Poisson
# F_pois = np.array([[1],
#                     [0]])
# G_pois = np.array([[1, 1], 
#                     [0, 1]])
F_pois = np.array([[1],
                    [0],
                    [1],
                    [0]])
G_pois = scipy.linalg.block_diag([[1, 1], 
                                [0, 1]],
                                [[np.cos(2*np.pi/period), np.sin(2*np.pi/period)],
                                [-np.sin(2*np.pi/period), np.cos(2*np.pi/period)]])

F_pois = np.r_[np.array([[1]]), F_pois]
G_pois = scipy.linalg.block_diag(1, G_pois)
delta_pois = 0.98 # Discount factor for evolution variance
RE_rho_pois = 0.99 # Discount factor for random effect. When = 1, no RE
# delta_pois = np.diag(np.concatenate((np.array([RE_rho_pois]), \
#                                       np.repeat(delta_pois, len(F_pois)-1)), axis=0))
delta_pois = scipy.linalg.block_diag(np.array([(1-RE_rho_pois)/RE_rho_pois]), \
                        np.matlib.repmat((1-delta_pois)/delta_pois,2,2),
                        np.matlib.repmat((1-delta_pois)/delta_pois,2,2))



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

nSample = 1000
pois_sample = np.zeros((nSample, TActual, N))
bern_sample = np.zeros((nSample, TActual, N))
st = time.time()
for n in range(N):
    # n = 0
    print(n)
    # Bern: Forward Filtering
    mN = m[:,np.where(unique_nodes == unique_edges[n,0])[0]]
    
    mt_bern, Ct_bern, at_bern, Rt_bern, rt_bern, st_bern, skipped_bern = \
    FF_Bernoulli2(F_bern, G_bern, delta_bern, flow_count, n, T0, TActual, eps)
    
      
    # Bern: Retrospective Analysis
    RA_prob = Retro_sampling2(TActual, F_bern, G_bern, mt_bern, \
                      Ct_bern, at_bern, Rt_bern, skipped_bern,\
                      nSample = nSample, family="bernoulli")
            
    
    # # Forward Filtering
    [mt_pois, Ct_pois, at_pois, Rt_pois, rt_pois, ct_pois, skipped_pois] = \
        FF_Poisson2(F_pois, G_pois, delta_pois, flow_count, n, mN, T0, TActual, \
                    eps, RE_rho_pois, conditional_shift_pois)
              
    
    RA_rate = Retro_sampling2(TActual, F_pois, G_pois, mt_pois, \
                      Ct_pois, at_pois, Rt_pois, skipped_pois,\
                      nSample = nSample, family="poisson")
           
        
    # Store them
    rt_bern_all[:, n] = rt_bern
    st_bern_all[:, n] = st_bern
    rt_pois_all[:, n] = rt_pois
    st_pois_all[:, n] = ct_pois

    # ssrt_bern_all[:, n] = ssrt_bern
    # ssst_bern_all[:, n] = ssst_bern
    # ssrt_pois_all[:, n] = ssrt_pois
    # ssst_pois_all[:, n] = ssct_pois
    bern_sample[:,:, n] = RA_prob
    pois_sample[:,:, n] = RA_rate
    