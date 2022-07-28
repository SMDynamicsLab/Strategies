# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:47:59 2022

@author: ASilva
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json


#%% Define Python user-defined exceptions.
class Error(Exception):
	"""Base class for other exceptions"""
	pass


#%% compute_asyn
# Compute asynchronies from stimulus and response time occurrences. Return a dataframe.
def Compute_Asyn(stim_times,resp_times):
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


#%% Error_Handling
# Function to errors handling. Return a tuple.
# asyn_df --> dataframe from function Compute_Asyn. resp_times --> list with the response times. max_stim_for_first_resp --> maximum number of stimulus for first response. 
def Error_Handling(asyn_df, resp_times, max_stim_for_first_resp):
	
	# Determine number of stimulus and responses registered.
	N_resp = len(resp_times)
	assigned_stim = asyn_df['assigned_stim'].values
	assigned_stim_NFR_filtered = []
	
	try: 
		if N_resp > 0: # If there were any response.

			# Find first stimulus with a decent response.
			first_assigned_stim = next(x for x in assigned_stim if x>=0)
			# If the first assigned stimulus doesn't much with any of the first stimulus, then re-do the trial.
			if first_assigned_stim >= max_stim_for_first_resp:			
				error_label = 'Error tipo NFR'
				error_type = 'NoFirstResp'
				raise Error 

			# Find non assigned responses.
			if any(assigned_stim==-1):
				error_label = 'Error tipo NAR'
				error_type = 'NonAssignedResp'
				raise Error

			# Find non assigned stimuli
			for i in range (len(assigned_stim)):
				if (assigned_stim[i] >= max_stim_for_first_resp):
					assigned_stim_NFR_filtered.append(assigned_stim[i])
			if (any(np.diff(assigned_stim_NFR_filtered)!=1) or assigned_stim_NFR_filtered[0] != max_stim_for_first_resp):
				error_label = 'Error tipo SS'
				error_type = 'SkipStim'
				raise Error

			# If the code got here, then the trial is valid!
			valid_trial = 1
			error_label = ''
			error_type = 'NoError'                    

		else: # If there were no responses.
			# Trial is not valid! then:
			error_label = 'Error tipo NR'
			error_type = 'NoResp'  
			raise Error

	except (Error):
		# Trial is not valid! then:
		valid_trial = 0
		
	return error_label, error_type, valid_trial


#%% Load_SubjectExpMetadata.
# Function to load experiment subject metadata. Return a dataframe.
# subject_number --> int (ej: 0). total_blocks --> int (ej: 1).
def Load_SubjectMetadata(subject_number, total_blocks):
	s_number = '{0:0>3}'.format(subject_number)
	conc_block_conditions_df = pd.DataFrame()
	for i in range(total_blocks):
		file_to_load = glob.glob('./Data/S'+s_number+"*-block"+str(i)+"-trials.csv")[0]
		file_to_load_df = pd.read_csv(file_to_load)
		conc_block_conditions_df = conc_block_conditions_df.append(file_to_load_df)	
	conc_block_conditions_df = conc_block_conditions_df.reset_index()
	conc_block_conditions_df = conc_block_conditions_df.drop(columns = ['index'])
	return conc_block_conditions_df


#%% Load_ExpMetadata.
# Function to load experiment metadata. Return a dataframe.
def Load_ExpMetadata(n_blocks):
	filename_names = './Data/Dic_names_pseud.dat'
	with open(filename_names) as f_names:
	   total_subjects = sum(1 for line in f_names) - 1
	f_names.close()
	conc_blocks_conditions_df = pd.DataFrame()
	for i in range(total_subjects):
		blocks_conditions_df = Load_SubjectMetadata(i, n_blocks)	
		conc_blocks_conditions_df = conc_blocks_conditions_df.append(blocks_conditions_df)
	conc_blocks_conditions_df = conc_blocks_conditions_df.reset_index()
	conc_blocks_conditions_df = conc_blocks_conditions_df.drop(columns = ['index'])
	conc_blocks_conditions_df.to_csv('./Data/ExpMetaData.csv')
	return conc_blocks_conditions_df


#%% Load_SingleTrial
# Function to load trial data from trial data file. Return a dataframe.
# subject_number --> int (ej: 0). block --> int (ej: 1). trial --> int (ej: 2).
def Load_SingleTrial(subject_number, block, trial):
	s_number = '{0:0>3}'.format(subject_number)
	file_to_load = glob.glob('./Data/S'+s_number+"*-block"+str(block)+"-trial"+str(trial)+".dat")[0]
	f_to_load = open(file_to_load,"r")
	content = f_to_load.read()
	f_to_load.close()
	content = json.loads(content)
	
	subject_n = []
	block_n = []
	trial_n = []
	event = []
	time = []
	order = []
	stim_times = []
	resp_times = []
	indexS = 0
	indexR = 0
	for i in range(len(content['Data'])):
		event.append(content['Data'][i][0:1])
		time.append(int(content['Data'][i][4:][:-1]))
		subject_n.append(subject_number)
		block_n.append(block)
		trial_n.append(trial)
		if (event[-1] == 'S'):
			order.append(indexS)
			stim_times.append(time[-1])
			indexS = indexS + 1
		if (event[-1] == 'R'):
			order.append(indexR)
			resp_times.append(time[-1])
			indexR = indexR + 1
	for i in range(len(content['Asynchrony'])):
		event.append('A')
		time.append(content['Asynchrony'][i])
		subject_n.append(subject_number)
		block_n.append(block)
		trial_n.append(trial)
		order.append(content['Stim_assigned_to_asyn'][i])
	
	trialData_df = pd.DataFrame()
	trialData_df = trialData_df.assign(Subject = subject_n, Block = block_n, Trial = trial_n, Event = event, Event_Order = order, Time = time)
	trialData_df.to_csv('./Data/TrialData.csv')
	return trialData_df


#%% Load_TrialsData
# Function to load trials data. Return a dataframe.
def Load_TrialsData(n_blocks):
	expMetadata_df = Load_ExpMetadata(n_blocks)
	conc_trialData_df = pd.DataFrame()
	for i in range(len(expMetadata_df.index)): 
		subject_number = expMetadata_df['Subject'][i]
		block = expMetadata_df['Block'][i]
		trial = expMetadata_df['Trial'][i]
		trialData_df = Load_SingleTrial(subject_number, block, trial)
		conc_trialData_df = conc_trialData_df.append(trialData_df)
	conc_trialData_df = conc_trialData_df.reset_index()
	conc_trialData_df = conc_trialData_df.drop(columns = ['index'])
	conc_trialData_df.to_csv('./Data/TrialsData.csv')
	return conc_trialData_df


#%% A look at the last trial.

# This function plots the stims and responses of a trial. Must be called from the console: Plot_RespStim(stim_time,resp_time).
def Plot_RespStim(stim_vector,resp_vector):
	N_stim = len(stim_vector)
	N_resp = len(resp_vector)

	my_labels = {"stim" : "Stimulus", "resp" : "Response"}
	for j in range(N_stim):
		plt.axvline(x=stim_vector[j],color='b',linestyle='dashed',label=my_labels["stim"])
		my_labels["stim"] = "_nolegend_"

	for k in range(N_resp):
		plt.axvline(x=resp_vector[k],color='r',label=my_labels["resp"])
		my_labels["resp"] = "_nolegend_"


	plt.axis([min(stim_vector)-100,max(resp_vector)+100,0,1])

	plt.xlabel('Tiempo[ms]',fontsize=12)
	plt.ylabel(' ')
	plt.grid()    
	plt.legend(fontsize=12)


#%% Trials_PerSubject_PerCondition.
# Function to load all trials per subject and per condition. Return a dataframe.
# subject --> int (ej: 0). condition --> int (ej: 1).
def Trials_PerSubject_PerCondition(subject, condition):
	
	# Filename for the file that contains all the experiment metadata.
	exp_metadata = './Data/ExpMetaData.csv'
	# Open ExpMetaData.csv file as dataframe.
	exp_metadata_df = pd.read_csv(exp_metadata)
	exp_metadata_df = exp_metadata_df.drop(columns = ['Unnamed: 0'])
	
	# Filename for the file that contains all the trials data.
	trials_data = './Data/TrialsData.csv'
	# Open TrialsData.csv file as dataframe.
	trials_data_df = pd.read_csv(trials_data)
	trials_data_df = trials_data_df.drop(columns = ['Unnamed: 0'])
	
	# Filter metadata by subject, condition and valid trial.
	exp_metadata_df = ((exp_metadata_df[(exp_metadata_df['Subject'] == subject) & (exp_metadata_df['Condition'] == condition) & (exp_metadata_df['Error'] == 'NoError')]).reset_index()).drop(columns = ['index'])
	
	# All trials per subject, per condition and per valid trial.
	allTrials_data_df = pd.DataFrame()
	for i in range (len(exp_metadata_df.index)):
		block = exp_metadata_df.loc[i,['Block']][0]
		trial = exp_metadata_df.loc[i,['Trial']][0]
		allTrials_data_df = allTrials_data_df.append(trials_data_df[(trials_data_df['Subject'] == subject) & (trials_data_df['Block'] == block) & (trials_data_df['Trial'] == trial)])
	allTrials_data_df = (allTrials_data_df.reset_index()).drop(columns = ['index'])
	
	allTrials_data_df.to_csv('./Data/AllTrialsPerSubjPerCond.csv')
	return allTrials_data_df


#%% Plot_Trials_Asyn_PerSubject_PerCondition.
# Function to plot all trials asynchronies per subject and per condition.
# subject --> int (ej: 0). condition --> int (ej: 1). figure_number --> int (ej: 1).
def Plot_Trials_Asyn_PerSubject_PerCondition(subject, condition, figure_number):
	allTrialsAsyn_data_df = Trials_PerSubject_PerCondition(subject, condition)
	allTrialsAsyn_data_df = allTrialsAsyn_data_df[allTrialsAsyn_data_df['Event'] == 'A']
	allBlocks = allTrialsAsyn_data_df['Block'].unique().tolist()
	
	for block in allBlocks:
		allTrialsAsyn_perBlock_data_df = allTrialsAsyn_data_df[allTrialsAsyn_data_df['Block'] == block]
		allTrials = allTrialsAsyn_perBlock_data_df['Trial'].unique().tolist()
		for trial in allTrials:
			allTrialsAsyn_perBlock_perTrial_data_df = allTrialsAsyn_perBlock_data_df[allTrialsAsyn_perBlock_data_df['Trial'] == trial]
			asyn_vector = allTrialsAsyn_perBlock_perTrial_data_df['Time'].tolist()
			Plot_asynch(asyn_vector, figure_number)


#%% Mean_Trials_Asyn_PerSubject_PerCondition.
# Function to calculate mean of trials asynchronies per subject and per condition. Return a list.
# subject --> int (ej: 0). condition --> int (ej: 1).
def Mean_Trials_Asyn_PerSubject_PerCondition(subject, condition):
	allTrialsAsyn_data_df = Trials_PerSubject_PerCondition(subject, condition)
	allTrialsAsyn_data_df = allTrialsAsyn_data_df[allTrialsAsyn_data_df['Event'] == 'A']
	allBlocks = allTrialsAsyn_data_df['Block'].unique().tolist()
	
	asyn_vector_df = pd.DataFrame()
	for block in allBlocks:
		allTrialsAsyn_perBlock_data_df = allTrialsAsyn_data_df[allTrialsAsyn_data_df['Block'] == block]
		allTrials = allTrialsAsyn_perBlock_data_df['Trial'].unique().tolist()
		for trial in allTrials:
			allTrialsAsyn_perBlock_perTrial_data_df = allTrialsAsyn_perBlock_data_df[allTrialsAsyn_perBlock_data_df['Trial'] == trial]
			asyn_vector = allTrialsAsyn_perBlock_perTrial_data_df['Time'].tolist()
			column = asyn_vector_df.shape[1]
			asyn_vector_df.insert(column, 'Trial' + str(column), asyn_vector)
	mean_trials_asyn = asyn_vector_df.T.mean().tolist()
	
	return mean_trials_asyn


#%% Plot_Mean_Trials_Asyn_AllSubjects_PerCondition.
# Function to plot mean trials asynchronies for all subjects and per condition.
# condition --> int (ej: 1). figure_number --> int (ej: 1).
def Plot_Mean_Trials_Asyn_AllSubjects_PerCondition(condition, figure_number):
	filename_names = './Data/Dic_names_pseud.dat'
	with open(filename_names) as f_names:
	   total_subjects = sum(1 for line in f_names) - 1
	f_names.close()

	for subject in range(total_subjects):
		mean_asyn_vector = Mean_Trials_Asyn_PerSubject_PerCondition(subject, condition)
		Plot_asynch(mean_asyn_vector, figure_number)


#%% Mean_Trials_Asyn_AllSubjects_AllConditions.
# Function to calculate mean of trials asynchronies for all subjects and for all conditions. Return a dataframe.
# n_subjects --> int (ej: 2). n_conditions --> int (ej: 3).
def Mean_Trials_Asyn_AllSubjects_AllConditions(n_subjects, n_conditions):
	asyn_vector_perCondition_df = pd.DataFrame()
	for condition in range(n_conditions):
		asyn_vector_df = pd.DataFrame()
		for subject in range(n_subjects):
			mean_asyn_vector = Mean_Trials_Asyn_PerSubject_PerCondition(subject, condition)
			column = asyn_vector_df.shape[1]
			asyn_vector_df.insert(column, 'Subj' + str(subject) + 'Cond' + str(condition), mean_asyn_vector)
		asyn_vector_perCondition_df.insert(condition, condition, asyn_vector_df.T.mean())
	
	return asyn_vector_perCondition_df


#%% Plot_Mean_Trials_Asyn_AllSubjects_AllConditions
# Function to plot mean trials asynchronies across all subjects and for all conditions.
# n_conditions --> int (ej: 3). figure_number --> int (ej: 1).
def Plot_Mean_Trials_Asyn_AllSubjects_AllConditions(n_conditions, figure_number):		
	filename_names = './Data/Dic_names_pseud.dat'
	with open(filename_names) as f_names:
	   total_subjects = sum(1 for line in f_names) - 1
	f_names.close()
	
	asyn_vector_df = Mean_Trials_Asyn_AllSubjects_AllConditions(total_subjects, n_conditions)
	total_columns = asyn_vector_df.shape[1]
	
	for column in range(total_columns):
		mean_asyn_vector = asyn_vector_df[asyn_vector_df.columns[column]].tolist()
		Plot_asynch(mean_asyn_vector, figure_number)


#%% Plot_asynch
# This function plots the asynchronys of a trial. Must be called from the console: Plot_asynch(asynchrony).
def Plot_asynch(asynch_vector, figure_number):
	plt.figure(figure_number)
	plt.plot(asynch_vector,'.-')
	plt.xlabel('# beep',fontsize=12)
	plt.ylabel('Asynchrony[ms]',fontsize=12)
	plt.grid() 

