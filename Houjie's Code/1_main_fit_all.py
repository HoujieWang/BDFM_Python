#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 15:08:52 2023

@author: wanghoujie
"""
# Load Directories for data and functions
import ast
import os
import numpy as np

# Prepare Data
full_data = PrepareData('flow.csv', 'occ.csv', 0.20)
locals().update(full_data)


# Setup Model structure

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
    # n = 1
    # Bern: Forward Filtering
    mN = m[:,np.where(categories == flowIndex[n,0])[0]]
    mt_bern, Ct_bern, at_bern, Rt_bern, rt_bern, st_bern, skipped_bern = FF_Bernoulli(F_bern, G_bern, delta_bern, flow, n, T0, TActual, eps)
    # Bern: Retrospective Analysis
    [sat_bern, sRt_bern, ssrt_bern, ssst_bern] = RA_Bernoulli(TActual, F_bern, G_bern, mt_bern, Ct_bern, at_bern, Rt_bern, skipped_bern)

    # Forward Filtering
    [mt_pois, Ct_pois, at_pois, Rt_pois, rt_pois, ct_pois, skipped_pois] = FF_Poisson(F_pois, G_pois, delta_pois, flow, n, mN, T0, TActual, eps, RE_rho_pois, conditional_shift_pois)
    # Retrospective Analysis
    [sat_pois, sRt_pois, ssrt_pois, ssct_pois] = RA_Poisson(TActual, F_pois, G_pois, mt_pois, Ct_pois, at_pois, Rt_pois, skipped_pois);
    
    # Store them
    rt_bern_all(:, n) = rt_bern;
    st_bern_all(:, n) = st_bern;
    rt_pois_all(:, n) = rt_pois;
    st_pois_all(:, n) = ct_pois;

    ssrt_bern_all(:, n) = ssrt_bern;
    ssst_bern_all(:, n) = ssst_bern;
    ssrt_pois_all(:, n) = ssrt_pois;
    ssst_pois_all(:, n) = ssct_pois;

    disp(n)
end

% Decompose fitted data
[fEst, fUpper, fLower, aiEst, aiUpper, aiLower, bjEst, bjUpper, bjLower, ...
    gijEst, gijUpper, gijLower] = ...
    Recouple_DGM2(rt_bern_all, st_bern_all, rt_pois_all, st_pois_all,...
    conditional_shift_pois, ...
    TActual, flowIndex, categories, 1000, I, N);

save('Output/Result_DGM_seq.mat', ...
    'fEst', 'fUpper', 'fLower', 'aiEst', 'aiUpper', 'aiLower', 'bjEst', ...
    'bjUpper', 'bjLower', ...
    'gijEst', 'gijUpper', 'gijLower')


[fEst_r, fUpper_r, fLower_r, aiEst_r, aiUpper_r, aiLower_r, bjEst_r, bjUpper_r, bjLower_r, ...
    gijEst_r, gijUpper_r, gijLower_r] = ...
    Recouple_DGM2(ssrt_bern_all, ssst_bern_all, ssrt_pois_all, ssst_pois_all,...
    conditional_shift_pois, ...
    TActual, flowIndex, categories, 2000, I, N);

save('Output/Result_DGM_retro.mat', ...
    'fEst_r', 'fUpper_r', 'fLower_r', ...
    'aiEst_r', 'aiUpper_r', 'aiLower_r', 'bjEst_r', ...
    'bjUpper_r', 'bjLower_r', ...
    'gijEst_r', 'gijUpper_r', 'gijLower_r')

% Get empirical decomposition
[mEf, mEai, mEbj, mEgij] = EmpiricalDecomp(flow, flowIndex, categories,...
    m, N, I, TActual, T0, eps);

save('Output/Result_DGM_empirical.mat', ...
    'mEf', 'mEai', 'mEbj', 'mEgij')

