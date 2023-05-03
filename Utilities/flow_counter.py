import numpy as np
import pandas as pd
from datetime import datetime


def flow_counter(loc_history, agent_id, region_bounds, cell_size, st_date, end_date, time_intl):
    # loc_history: a data frame of all agent activity (check rio bus data as an example ) 
    # agent_id: an array of names of indices
    # region_bounds: an array specifying the binding box of length 4 [ymin, xmin, ymax, xmax]
    # cell_size: cell_size*111 kilometers is the side of each spatial grid cell
    # st_date: a string indiating the starting date (e.g. '02-22-2019')
    # end_date: a string indiating the ending date
    # time_intl:  a number specifying the observation interval in the unit of minutets 
    
    # Temporal part
    st_date = datetime.strptime(st_date, '%m-%d-%Y')
    end_date = datetime.strptime(end_date, '%m-%d-%Y')
    duration = end_date - st_date
    time_grid = pd.date_range(st_date, \
                              periods=1440 / time_intl * (duration.days + 1),\
                              freq=str(time_intl)+'min')
    
    # Spatial part
    lon_coords = np.arange(region_bounds[0], region_bounds[2]+cell_size, cell_size) 
    lat_coords = np.arange(region_bounds[1], region_bounds[3]+cell_size, cell_size)
    spatial_grid = np.array(np.meshgrid(lon_coords, lat_coords)).\
        reshape(2, len(lon_coords)*len(lat_coords)).T
    
    # We initialize it to be -1
    agent_location = (np.zeros((len(agent_id), len(time_grid)-1))-1).astype("int")
    agent_time = np.array([np.datetime64(datetime.strptime(t, '%m-%d-%Y%H:%M:%S')) \
                            for t in np.array(loc_history.date) + np.array(loc_history.time)])
    
    for i in range(len(time_grid)-1):
        # i = 0
        t0 = time_grid[i]
        t1 = time_grid[i+1]
        loc_history_tmp = loc_history.iloc[(t0 < agent_time) & (agent_time < t1), ]
        for j in range(len(agent_id)):
            # bus = 'D13022'
            bus = agent_id[j]
            if (sum(loc_history_tmp.order == bus) != 0):
                loc_history_tmp_i = loc_history_tmp.iloc[np.array(loc_history_tmp.order == bus), :]
                # the point is the largest lat,long point that are smaller than 
                # the bus's latest lat,long point in the current time interval
                lon_point = lon_coords[np.where(lon_coords > loc_history_tmp_i.longitude.iloc[-1])[0][0]-1]
                lat_point = lat_coords[np.where(lat_coords > loc_history_tmp_i.latitude.iloc[-1])[0][0]-1]
                
                index = np.where((spatial_grid[:,0] == lon_point) & (spatial_grid[:,1] == lat_point))[0][0]
                agent_location[j, i] = index
    
    for t in np.arange(1,agent_location.shape[1]): 
        # t = 1
        temp_idx = (agent_location[:,t] == -1) & (agent_location[:,(t-1)] != -1)
        agent_location[temp_idx, t] = agent_location[temp_idx, (t-1)]
    
    
    
    edge_idx = [None] * (agent_location.shape[1]-1)
    last_location = agent_location[:, 0]
    for i in np.arange(1,agent_location.shape[1]): 
        # i = 1
        
        # create edge labels based on flows
        agent_location_i = np.vstack([last_location, agent_location[:, i]]).T
        
        # last_location == -1 means we haven't observe a location for this agent yet
        edgeIndex_tmp = agent_location_i[last_location != -1, ] 
        
        #
        last_location = agent_location_i[:, 1]
        
        edge_idx[i-1] = edgeIndex_tmp
    
    
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
    
    return agent_location, flow_count, unique_edges, unique_nodes, agent_id, m    