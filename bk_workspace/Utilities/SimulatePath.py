import numpy as np

def SimulatePath(start_location, start_time, end_time, bernoulli_mean, poisson_mean, 
                 unique_edges):
    path = np.zeros(end_time - start_time)
    i = 0
    possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == start_location]
    for t in np.arange(start_time, end_time):
        Z = [np.random.binomial(1, bernoulli_mean[t, n]) for n in possible_steps]
        # We need a better way to handle the case when all Zs are zero.
        #############################################################
        if all(z == 0 for z in Z):
            path[i] = location
            i = i + 1
            continue
        ##############################################################
        z_possible = [n for n, z in zip(realized_edges, Z) if z == 1]
        rates = poisson_mean[:, n][z_possible]
        probs = rates / sum(rates)
        location_index = np.random.choice(z_possible, 1, True, probs)
        location = unique_edges[location_index, 1]
        path[i] = location
        i = i + 1
        possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    return(path)