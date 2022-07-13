# Compute asynchronies from stimulus and response time occurrences

import numpy as np
import pandas as pd

def compute_asyn(stim_times,resp_times):
	if len(resp_times)==0:
		asyn = []
		assigned_stim = []
	else:
		ISI = np.median(np.diff(stim_times))
		asyn_max = round(ISI/2)
		n_stims = len(stim_times)
		n_resps = len(resp_times)
		minima_R = np.zeros((n_stims,n_resps),dtype=int)
		minima_S = np.zeros((n_stims,n_resps),dtype=int)
		assigned_stim = -np.ones(n_resps,dtype=int)
		stimuli_flag = np.zeros(n_stims,dtype=int)
		asyn = np.full(n_resps,np.nan)
		asyn_max = ISI/2
	
		# Find matching S-R pairs
	
		# pairwise differences between every response and every stimulus
		# (dimensions = number of stimuli x number of responses)
		differences = -np.subtract.outer(stim_times,resp_times).astype(float)	# type float so it can be NaN
		differences[abs(differences)>=asyn_max] = np.nan # remove differences larger than threshold
	
		# for every response, find the closest stimulus (nontrivial if more responses than stimuli)
		# IM SURE THIS LOOP CAN BE VECTORIZED
		for resp in range(n_resps):
			aux = differences[:,resp]
			# prevent "no non-missing arguments to min; returning Inf" warning
			#if np.any(np.isnan(aux)): # find at least one non-missing value
			min_abs = np.nanmin(abs(aux))
			min_idx = np.where(abs(aux)==min_abs)
			minima_R[min_idx,resp] = 1
	
	
		# remove multiple responses closest to a single stimulus (row-wise consecutive 1's)
		# i.e. make no attempt at deciding
		minima_shift_R = minima_R + np.roll(minima_R,(0,1),(0,1)) + np.roll(minima_R,(0,-1),(0,1))
		minima_R[np.where(minima_shift_R>=2)] = 0
	
		# for every stimulus, find the closest response (nontrivial if more stimuli than responses)
		# IM SURE THIS LOOP CAN BE VECTORIZED
		for stim in range(n_stims):
			aux = differences[stim,:]
			# prevent "no non-missing arguments to min; returning Inf" warning
			#if np.any(np.isnan(aux)): # find at least one non-missing value
			min_abs = np.nanmin(abs(aux))
			min_idx = np.where(abs(aux)==min_abs)
			minima_S[stim,min_idx] = 1
	
		# remove multiple stimuli closest to a single response (col-wise consecutive 1's)
		# i.e. make no attempt at deciding
		minima_shift_S = minima_S + np.roll(minima_S,(1,0),(0,1)) + np.roll(minima_S,(-1,0),(0,1))
		minima_S[np.where(minima_shift_S>=2)] = 0
	
		# matching pairs are represented by intersections (i.e. common 1's)
		minima_intersect = minima_R*minima_S
	
		# save asynchronies: get row and column for every matched pair
		SR_idxs = np.where(minima_intersect==1)
		S_idx = SR_idxs[0]
		R_idx = SR_idxs[1]
	
		# keep track of which stimulus was assigned to each response (-1 if not assigned)
		assigned_stim[R_idx] = S_idx
		# keep track of assigned stimuli (0 if not assigned)
		stimuli_flag[S_idx] = np.ones(len(S_idx))
	
		# save asynchrony (NaN means that response was not assigned)
		if (S_idx.size != 0 and R_idx.size != 0):
			lin_ind = np.ravel_multi_index((S_idx,R_idx),differences.shape)
			asyn[R_idx] = differences.ravel()[lin_ind]
	
	# Output
	output = pd.DataFrame({'asyn':asyn,'assigned_stim':assigned_stim})
	return output
