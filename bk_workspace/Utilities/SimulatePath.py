import numpy as np

""" bernoulli_mean = ra_bern_mean
poisson_mean = ra_pois_mean
start_time = 0
end_time = TActual
location = 26 """

def SimulatePath(location, start_time, end_time, bernoulli_mean, poisson_mean, 
                 unique_edges):
    N = unique_edges.shape[0]
    path = np.zeros(end_time - start_time + 1)
    i = 0
    path[i] = location
    possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    for t in np.arange(start_time, end_time):
        i = i + 1
        Z = [np.random.binomial(1, bernoulli_mean[t, n]) for n in possible_steps]
        # We need a better way to handle the case when all Zs are zero.
        #############################################################
        if all(z == 0 for z in Z):
            path[i] = location
            continue
        z_possible = [n for n, z in zip(possible_steps, Z) if z == 1]
        rates = poisson_mean[t, :][z_possible]
        probs = rates / sum(rates)
        location_index = np.random.choice(z_possible, 1, True, probs)
        location = unique_edges[location_index, 1]
        path[i] = location
        possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    return(path)

""" Z = np.ones(len(possible_steps)) """