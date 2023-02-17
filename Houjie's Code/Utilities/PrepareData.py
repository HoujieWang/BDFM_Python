import pandas as pd
import numpy as np
def PrepareData(fname_flow, fname_occ, pct_time_w_traffic):
# =============================================================================
#     INPUT
#     Two parameters specify file names of data, stored at "Data/"
#     fname_flow: File name of the flow data: ((2 + Num_pairs) * T)
#                  First two columns are two node IDs
#                  Num_pairs: number of pairs
#                  T: Number of time
# 
#     fname_occ: File name of occ, Count of total outflow from each node
#                 at each time period. (Num of node * T)
#     
#     OUTPUT
#     flow: cleaned flow data (removed node IDs)
#     flowIndex: store IDs of node pairs
#     occ: outflow traffic of unique nodes per time
#     categories: Node IDs
#     T0: Number of leading time periods to exclude (for p.5 Eq (2))
#     TActual: Actual number of time periods
# =============================================================================
    
#   fname_flow = 'flow.csv'
#   fname_occ = 'occ.csv'
#   pct_time_w_traffic = 0.2
    
    # Load data and slight clean
    flowRaw    = pd.read_csv(fname_flow, header=None).to_numpy(dtype = "double")
    occRaw     = pd.read_csv(fname_occ, header=None).to_numpy(dtype = "double")

    flowIndex  = flowRaw[:, 0: 2]
    flow       = flowRaw[:, 2: flowRaw.shape[1]]

    categories = occRaw[:, 0]
    occ        = occRaw[:, 1: occRaw.shape[1]].T
    
    # Specify parameters describing dimensions of data
    
    # Number of unique nodes (including (external))
    I = occ.shape[1]
    # Number of pairs (node_i --> node_j)
    N = flow.shape[0]
    
    # Time
    TTotal  = occ.shape[0]
    T0 = 2 # Periods to skip
    TActual = TTotal - T0
    
    # Threshold for dirichlet parameter & gamma shape parameter
    eps = 1e-2
    flow[flow == 0] = eps

    # m
    # Calculate Occupancy correction ratea (Chen, Banks & West, 2019, p.5)
    m = np.zeros((TActual,I), dtype = 'double')
    m[:, (I-1)] = 1
    for t in range(TActual):
        for i in range(I):
            if occ[t+T0,i] == 0:
                m[t,i] = 0;
            elif occ[t+T0, i] <= 10 or occ[t+T0-1, i] <= 10:
                m[t, i] = 1
            else:
                m[t, i] = occ[t+T0, i]/occ[t+T0-1, i]
            
        
    
    mVector = m.copy()
    mLower = np.quantile(mVector, 0.1)
    mUpper = np.quantile(mVector, 0.9)

    # Remove flows with insufficient traffic
    if pct_time_w_traffic < 1:
        mS  = np.sum(flow > eps, 1)
        mS  = np.where(mS > pct_time_w_traffic * TTotal)[0]
 
        flow = flow[mS, :]
        flowIndex = flowIndex[mS, ]
        N = flow.shape[0] # reset N
        
    out = {
        "flow": flow,
        "flowIndex": flowIndex,
        "occ": occ,
        "categories": categories,
        "T0": T0,
        "TActual": TActual,
        "I": I,
        "N": N,
        "eps": eps,
        "m": m,
        "mLower": mLower,
        "mUpper": mUpper
        }
    return out
    


