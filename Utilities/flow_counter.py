import numpy as np
import pandas as pd
from datetime import datetime
import scipy

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
                              periods= 1440 / time_intl * (duration.days + 1) + 1,\
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

    
def flow_counter2(loc_history, agent_id, region_bounds, cell_size, st_date, end_date, time_intl):
    # loc_history = data; agent_id = ['167']; region_bounds = region_bounds
    # cell_size=0.01,; st_date = '03-01-2008'; end_date = '03-14-2008'
    # Temporal part
    st_date = datetime.strptime(st_date, '%m-%d-%Y')
    end_date = datetime.strptime(end_date, '%m-%d-%Y')
    duration = end_date - st_date
    time_grid = pd.date_range(st_date, \
                              periods= 1440 / time_intl * (duration.days + 1) + 1,\
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
        # i = 12
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
    
    for i in range(agent_location.shape[0]):
        # i = 0
        temp_idx = np.where(agent_location[i, :] != -1)[0][0]
        agent_location[i, 0: temp_idx] = agent_location[i, temp_idx]
    
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
    dist_mat = scipy.spatial.distance_matrix(spatial_grid[unique_nodes,:], \
                                             spatial_grid[unique_nodes,:], \
                                                 p=2)
    return agent_location, flow_count, unique_edges, unique_nodes, agent_id, m, dist_mat    


def flow_converger(data, nzones, distance = False, dist_mat=[], reverse = False):
    # data = agent_location_i; nzones = 10
    unique_nodes_i = np.unique(data)
    flow_i = np.hstack((data[: -1, :], data[1:  , :]))
    unique_edges_i = np.unique(flow_i, axis = 0)
    flow_count_i = np.zeros((unique_edges_i.shape[0], data.shape[0]-1))
    for ii in range(unique_edges_i.shape[0]):
        flow_count_i[ii, np.sum(np.abs(flow_i - unique_edges_i[ii, :]), axis=1) == 0] = 1
    
    node_order = np.zeros((unique_nodes_i.shape[0], nzones+1))-1
    node_order[:,0] = unique_nodes_i
    
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
            if reverse:
                end_nodes = end_nodes[np.arange(len(end_nodes)-1, -1, -1)]
            node_choices[:len(end_nodes)] = end_nodes
        
        node_order[i,1:] = node_choices
    

    trans_flow_i = np.zeros((node_order.shape[1]-1, len(data)-1))-1
    # temp = np.hstack((data[:-1,:], data[1:,:]))
    for t in np.arange(len(data)-1):
        # t = 48
        last_node = data[t,0]
        now_node = data[t+1,0]
        now_node_rank = np.where(node_order[node_order[:,0] == last_node, 1:] == now_node)[1][0]
        binary_y = np.array([1]*now_node_rank + [0])
        trans_flow_i[:len(binary_y), t] = binary_y
    return trans_flow_i, node_order.astype(int)






