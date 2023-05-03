import numpy as np
import itertools
import pandas as pd

def SimulatePath(start_location, start_time, end_time, bern_samps, pois_samps, 
                 unique_edges):
    sims = bern_samps.shape[0]
    N = unique_edges.shape[0]
    paths = np.zeros((sims, end_time - start_time + 1))
    for s in np.arange(0, sims):
        i = 0
        location = start_location
        paths[s, i] = location
        possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
        for t in np.arange(start_time, end_time):
            i = i + 1
            Z = [np.random.binomial(1, bern_samps[s, t, n]) for n in possible_steps]
            # We need a better way to handle the case when all Zs are zero.
            #############################################################
            if all(z == 0 for z in Z):
                paths[s, i] = location
                continue
            z_possible = [n for n, z in zip(possible_steps, Z) if z == 1]
            rates = pois_samps[s, t, :][z_possible]
            probs = rates / sum(rates)
            location_index = np.random.choice(z_possible, 1, True, probs)
            location = unique_edges[location_index, 1]
            paths[s, i] = location
            possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    return(paths)

def SimulatePath2(start_location, start_time, end_time, bern_mean, pois_mean, 
                 unique_edges, sims):
    N = unique_edges.shape[0]
    paths = np.zeros((sims, end_time - start_time + 1))
    for s in np.arange(0, sims):
        i = 0
        location = start_location
        paths[s, i] = location
        possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
        for t in np.arange(start_time, end_time):
            i = i + 1
            Z = [np.random.binomial(1, bern_mean[t, n]) for n in possible_steps]
            # We need a better way to handle the case when all Zs are zero.
            #############################################################
            if all(z == 0 for z in Z):
                paths[s, i] = location
                continue
            z_possible = [n for n, z in zip(possible_steps, Z) if z == 1]
            rates = pois_mean[t, :][z_possible]
            probs = rates / sum(rates)
            location_index = np.random.choice(z_possible, 1, True, probs)
            location = unique_edges[location_index, 1]
            paths[s, i] = location
            possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    return(paths)


""" t= 0
senders = np.unique(unique_edges[:,0])
sender = senders[0]
receivers = [n for n in np.arange(0, N) if unique_edges[n, 0] == sender]
sims = 1000
steps = np.zeros(sims)
for s in np.arange(0, sims):
    Z = [np.random.binomial(1, bern_mean[t, n]) for n in receivers]
    if all(z == 0 for z in Z):
        steps[s] = sender
        continue
    z_possible = [n for n, z in zip(receivers, Z) if z == 1]
    rates = pois_mean[t, :][z_possible]
    probs = rates / sum(rates)
    location_index = np.random.choice(z_possible, 1, True, probs)
    steps[s] = unique_edges[location_index, 1]
pd.Categorical(steps, categories = unique_edges[receivers, 1]) """
""" Z = np.ones(len(possible_steps)) """

def GetTransitionProbs(unique_edges, bern_mean, pois_mean, TActual, N):
    senders = np.unique(unique_edges[:,0])
    transition_probs = np.zeros((TActual, N))

    for sender in senders:
        receivers = [n for n in np.arange(0, N) if unique_edges[n, 0] == sender]

        all_z = [[0, 1]] * len(receivers)
        all_z = list(itertools.product(*all_z))
        all_z1 = pd.DataFrame(all_z).drop(0)

        all_z0 = 1 - all_z1

        for t in np.arange(0, TActual):
            bern_probs = bern_mean[t, :][receivers]

            Z1_probs = (all_z1.apply(lambda x: x * np.log(bern_probs), axis = 1)
            .apply(lambda x: np.sum(x), axis = 1)
            #.apply(lambda x: np.exp(x))
            )

            Z0_probs = (all_z0.apply(lambda x: x * np.log(1 - bern_probs), axis = 1)
            .apply(lambda x: np.sum(x), axis = 1)
            #.apply(lambda x: np.exp(x))
            )

            Z_probs = np.exp(Z1_probs + Z0_probs)
            Z_probs = Z_probs/sum(Z_probs)

            pois_rates = pois_mean[t, :][receivers]
            pois_probs = (all_z.apply(lambda x: x * pois_rates, axis = 1)
            .apply(lambda x: x / sum(x), axis = 1)
            )

            integrate_z = (pois_probs.apply(lambda x: Z_probs * x, axis = 0)
            .apply(lambda x: sum(x), axis = 0)
            )

            transition_probs[t, receivers] = integrate_z

    return(transition_probs)

bus_location[8,:]