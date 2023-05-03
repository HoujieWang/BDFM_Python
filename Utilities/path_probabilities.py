import numpy as np
import pandas as pd
import itertools

""" At the moment, these functions use posterior means of the Bernoulli
and Poisson retrospective distributions. Changing "simulate_path" for use with the
retrospective samples should be easy. Changing the "transition_probs" function might be harder. 
 """

def simulate_path(start_location, start_time, end_time, bern_mean, pois_mean, 
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

            # Condition away case where all Z are 0
            Z = np.zeros(len(possible_steps))
            while all(z == 0 for z in Z):
                Z = [np.random.binomial(1, bern_mean[t, n]) for n in possible_steps]

            z_possible = [n for n, z in zip(possible_steps, Z) if z == 1]
            rates = pois_mean[t, :][z_possible]
            probs = rates / sum(rates)
            location_index = np.random.choice(z_possible, 1, True, probs)
            location = unique_edges[location_index, 1]
            paths[s, i] = location
            possible_steps = [n for n in np.arange(0, N) if unique_edges[n, 0] == location]
    return(paths)

def trasition_probs_exact(unique_edges, bern_mean, pois_mean, TActual, N):
    senders = np.unique(unique_edges[:,0])
    transition_probs = np.zeros((TActual, N))

    for sender in senders:
        receivers = [n for n in np.arange(0, N) if unique_edges[n, 0] == sender]

        all_z = [[0, 1]] * len(receivers)
        all_z = list(itertools.product(*all_z))

        # Remove case when z = 0 for all paths
        all_z1 = pd.DataFrame(all_z).drop(0)

        all_z0 = 1 - all_z1

        for t in np.arange(0, TActual):
            bern_probs = bern_mean[t, :][receivers]

            # Gathers p from bern_mean
            Z1_probs = (all_z1.apply(lambda x: x * np.log(bern_probs), axis = 1)
            .apply(lambda x: np.sum(x), axis = 1)
            )

            # Gathers (1 - p) from bern_mean
            Z0_probs = (all_z0.apply(lambda x: x * np.log(1 - bern_probs), axis = 1)
            .apply(lambda x: np.sum(x), axis = 1)
            )

            # Combine and renormalize
            Z_probs = np.exp(Z1_probs + Z0_probs)
            Z_probs = Z_probs/sum(Z_probs)

            # Transition probs conditional on each possible Z
            pois_rates = pois_mean[t, :][receivers]
            pois_probs = (all_z1.apply(lambda x: x * pois_rates, axis = 1)
            .apply(lambda x: x / sum(x), axis = 1)
            )

            # Average over possible Z
            integrate_z = (pois_probs.apply(lambda x: Z_probs * x, axis = 0)
            .apply(lambda x: sum(x), axis = 0)
            )

            transition_probs[t, receivers] = integrate_z

    return(transition_probs)

def trasition_probs_sim(unique_edges, bern_mean, pois_mean, TActual, N, sims):
    senders = np.unique(unique_edges[:,0])
    transition_probs = np.zeros((TActual, N))

    for sender in senders:
        receivers = [n for n in np.arange(0, N) if unique_edges[n, 0] == sender]

        for t in np.arange(0, TActual):
            steps = np.zeros(sims)

            for s in np.arange(0, sims):
                # Condition away case where all Z are 0
                Z = np.zeros(len(receivers))
                while all(z == 0 for z in Z):
                    Z = [np.random.binomial(1, bern_mean[t, n]) for n in receivers]

                z_possible = [n for n, z in zip(receivers, Z) if z == 1]
                rates = pois_mean[t, :][z_possible]
                probs = rates / sum(rates)
                location_index = np.random.choice(z_possible, 1, True, probs)
                location = unique_edges[location_index, 1]
                steps[s] = location
            thing = pd.Categorical(steps, categories=unique_edges[receivers, 1])
            probs = thing.value_counts()/sims
            transition_probs[t, receivers] = probs
    return(transition_probs)