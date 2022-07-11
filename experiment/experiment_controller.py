# -*- coding: utf-8 -*-
"""
Created on Jun 2022

@author: paulac, ArielSilva
"""

import serial, time
import numpy as np
import numpy.matlib as mlib
import matplotlib.pyplot as plt
import random
import os
import glob
import pandas as pd
import json


#%% Description

#==============================================================================
# Saves:  - a file per trial containing the raw data from it.
#         - a file per trial containing extracted data from it.
#         - a file per block containing information about all trials in it.
#==============================================================================


#%% Definitions.

# Define variables.

ISI = 500                     	# Interstimulus interval (milliseconds).
n_stim = 10                     # Number of bips within a sequence.
n_trials_percond = 4			# Number of trials per condition.
n_blocks = 3                 	# Number of blocks.
n_subj_max = 100             	# Maximum number of subjects.
perturb_type = [2,3]      # Perturbation type. 0--> No perturbation. 1--> Step change. 2--> Phase shift.
perturb_size = 100           	# Perturbation size.
perturb_bip_range = (5,7)		# Perturbation bip range.


#%% Define Python user-defined exceptions.
class Error(Exception):
	"""Base class for other exceptions"""
	pass


#%% Conditions

# Filename for the file that will contain all possible permutations for the subjects.
presentation_orders = './Data/Presentation_orders.csv'

# All possible conditions for perturbations.
all_conditions = ['pos', 'neg', 'iso']
n_conditions = len(all_conditions)      # Number of conditions.

# Condition dictionary so we can choose the condition without going through number position.
condition_dictionary = {"pos": 0,"neg": 1,"iso": 2}

# Number of trials.
n_trials = n_trials_percond * n_conditions
if (n_trials % n_blocks == 0):
	n_trials_perblock = n_trials // n_blocks 
else:
	print('Error: Número de trials no es múltiplo del número de bloques.')
	raise Error

# Start experiment or generate Perturbation_orders.csv file.
start_or_generate_response = input("Presione enter para iniciar experimento, o escriba la letra G (generar archivo con órdenes de presentación) y presione enter: ") 

# If this is the first time running the experiment, then it's necessary to generate the Presentation_orders.csv file.
if start_or_generate_response == 'G':
	confirm_response = input('¿Está seguro? Si el archivo ya existe se sobrescribirá. Escriba S y presione enter para aceptar, o sólo presione enter para cancelar: ')
	
	if confirm_response == 'S':
		chosen_conditions = mlib.repmat(np.arange(0,n_conditions),n_subj_max,(n_trials_percond))
		for i in range(0,n_subj_max):
			random.shuffle(chosen_conditions[i])

		presentation_orders_df = pd.DataFrame()
		
		for i in range(0,n_subj_max):
			next_subject_number = '{0:0>3}'.format(i)
			next_subject_number = 'S' + next_subject_number
			presentation_orders_df[next_subject_number] = chosen_conditions[i]
        
		presentation_orders_df.index.name="Trial"

		presentation_orders_df.to_csv(presentation_orders)


else:

	#Communicate with arduino.
	arduino = serial.Serial('COM4', 9600)
	   
	# Open Presentation_orders.csv file as dataframe.
	presentation_orders_df = pd.read_csv(presentation_orders,index_col='Trial')
	
	
	# EXPERIMENT

	# Check for file with names and pseudonyms.
	filename_names = './Data/Dic_names_pseud.dat'
	
	cont_exp = input('\nSi necesita retomar el último experimento, escriba S y presione enter. De lo contrario, presione enter para continuar: ')
	if (cont_exp == 'S'):
		try:
			f_names = open(filename_names,"r")
			
			content = f_names.read()
			last_subject_number = int(content [-3:])
			curr_subject_number = '{0:0>3}'.format(last_subject_number)
			f_names.close()
		except IOError:
			print('El archivo no esta donde debería, o no existe un sujeto previo.')
			raise
		
		current_block_counter = input('\nIngrese el número de bloque deseado y presione enter. De lo contrario, presione enter para seleccionar 0 por defecto: ')
		if (current_block_counter != ""):
			if (int(current_block_counter) >= 0 and int(current_block_counter) < n_blocks):
				# Run blocks.
				block_counter = int(current_block_counter)
			else:
				print('El número de bloque seleccionado no es válido.')
				raise Error
		else:
			# Run blocks.
			block_counter = 0
		
		
	else:
		try:
			f_names = open(filename_names,"r")

			if os.stat(filename_names).st_size == 0:
				curr_subject_number_int = 0
				curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
				f_names.close()
			else:
				content = f_names.read()
				last_subject_number = int(content [-3:])
				curr_subject_number_int = last_subject_number + 1
				curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
				f_names.close()
            
		except IOError:
			print('El archivo no está donde debería, ubicarlo en la carpeta correcta y volver a correr esta celda.')
			raise
			
		# Set subject name for filename.
		name = input("Ingrese su nombre: ") 
		f_names = open(filename_names,"a")
		f_names.write('\n'+name+'\tS'+curr_subject_number)
		f_names.close()

	    # Run blocks.
		block_counter = 0


	# Trials for the current subject.
	subject_df = pd.DataFrame(presentation_orders_df['S' + curr_subject_number])
	subject_df.rename(columns={'S' + curr_subject_number:'Condition'},inplace=True)


	while (block_counter < n_blocks):

		# Block conditions.
		block_conditions_aux = block_counter * n_trials_perblock
		block_conditions_df = (subject_df.loc[block_conditions_aux : block_conditions_aux + n_trials_perblock - 1]).reset_index()
		block_conditions_df = block_conditions_df.drop(columns = ['Trial'])
		block_conditions_df.index.name="Trial"
		block_counter_list = []
		perturb_bip_list = []
		perturb_size_list = []
		perturb_type_list = []
		for i in range(0,n_trials_perblock):
			block_counter_list.append(block_counter)
			perturb_bip_list.append(random.randrange(perturb_bip_range[0],perturb_bip_range[1],1))
			trial_type = (block_conditions_df.loc[[i]].values.tolist())[0][0]
			if (trial_type == 0):
				perturb_size_list.append(perturb_size)
			elif (trial_type == 1):
				perturb_size_list.append(-perturb_size)
			else:
				perturb_size_list.append(0)
			perturb_type_list.append(random.choice(perturb_type))
					
		block_conditions_df = block_conditions_df.assign(Block = block_counter_list, Original_trial = range(0,n_trials_perblock), 
            Perturb_bip = perturb_bip_list, Perturb_size = perturb_size_list, Perturb_type = perturb_type_list)
		block_conditions_df = block_conditions_df.reindex(columns=['Block','Original_trial','Condition','Perturb_bip','Perturb_size','Perturb_type'])


		# Run one block.
		input("Presione Enter para comenzar el bloque (%d/%d): " % (block_counter,n_blocks-1))
		
		# Set time for file name.
		timestr = time.strftime("%Y_%m_%d-%H.%M.%S")

		trial = 0
		
		messages = [] # Vector that will contain exact message sent to arduino to register the conditions played in each trial.
		valid_trials = [] # Vector that will contain 1 if the trial was valid or 0 if it wasn't.
		errors = [] # Vector that will contain the type of error that ocurred if any did.    
        
        # Generate filename for file that will contain all conditions used in the trial along with the valid_trials vector.
		filename_block = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-trials.csv"
        
		while (trial < len(block_conditions_df.index)):
			input("Presione Enter para comenzar el trial (%d/%d):" % (trial,len(block_conditions_df.index)-1))
			plt.close(1)
			plt.close(2)
            
            # Generate raw data file.
			filename_raw = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)+"-raw.dat"
			f_raw = open(filename_raw,"w+")
         
            # Generate extracted data file name (will save raw data, stimulus time, feedback time and asynchrony).
			filename_data = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)+".dat"    
            
            # Wait random number of seconds before actually starting the trial.
			wait = random.randrange(10,20,1)/10.0
			time.sleep(wait)
            
            # Define stimulus and feedback condition for this trial.
			perturb_size_aux = (block_conditions_df.loc[[trial],['Perturb_size']].values.tolist())[0][0]
			perturb_bip_aux = (block_conditions_df.loc[[trial],['Perturb_bip']].values.tolist())[0][0]
			perturb_type_aux = (block_conditions_df.loc[[trial],['Perturb_type']].values.tolist())[0][0]

			# Send message with conditions to arduino.
			message = str.encode(";S%c;F%c;N%c;A%d;I%d;n%d;P%d;B%d;T%d;X" % ('B', 'B', 'B', 3, ISI, n_stim, perturb_size_aux, perturb_bip_aux, perturb_type_aux))
			arduino.write(message)
			messages.append(message.decode())

			# Read information from arduino.
			data = []
			aux = arduino.readline().decode()
			while (aux[0]!='E'):
				data.append(aux)
				f_raw.write(aux) # save raw data
				aux = arduino.readline().decode()

			# Separates data in type, number and time.
			e_total = len(data)
			e_type = []
			e_number = []
			e_time = []
			for event in data:
				e_type.append(event.split()[0])
				e_number.append(int(event.split()[1]))
				e_time.append(int(event.split()[2]))

			# Separates number and time according to if it comes from stimulus or response.
			stim_number = []
			resp_number = []
			stim_time = []
			resp_time = []
			for events in range(e_total):
				if e_type[events]=='S':
					stim_number.append(e_number[events])
					stim_time.append(e_time[events])

				if e_type[events]=='R':
					resp_number.append(e_number[events])
					resp_time.append(e_time[events])

			# Determine number of stimulus and responses registered.
			N_stim = len(stim_time)
			N_resp = len(resp_time)

			# Close raw data file.    
			f_raw.close()

			# ---------------------------------------------------------------
			# Asynchronies calculation.

			# Vector that will contain asynchronies if they are calculated.
			asynchrony = []
			
			try: 
				if N_resp > 0: # If there were any responses.

					j = 0 # Stimulus counter.
					k = 0 # Responses counter for finding first stimuli with decent response.
					i = N_resp-1 # Responses counter for finding last stimuli with response.
					first_stim_responded_flag = False # Flag if there was a stimuli with a recent response.
					last_resp_flag = False                


					# Find first stimulus with a decent response.
					while j < 5: # If the first response doesn't match with any of the 5 first stimuli, then re-do the trial.
						diff = stim_time[j]-resp_time[k]
						if abs(diff)<200:
							first_stim_responded_index = j
							first_stim_responded_flag = True
							break
						else:
							j = j+1


					if first_stim_responded_flag == True:
						pass
					else:
						print('Error tipo NFR')
						errors.append('NoFirstResp')
						raise Error 


					# Find response to last stimulus (last response that should be considerated).
					while i > 0:
						diff = stim_time[N_stim-1]-resp_time[i]
						if abs(diff)<200:
							last_resp_index = i
							last_resp_flag = True
							break
						else:
							i = i-1
					
					if last_resp_flag == True:
						pass
					else:
						print('Error tipo NLR')
						errors.append('NoLastResp')
						raise Error 


					# New vectors of stimulus and responses that only contain those that have a pair of the other type.      
					stim_paired = stim_time[first_stim_responded_index:N_stim]
					resp_paired = resp_time[0:(last_resp_index+1)]
					N_stim_paired = len(stim_paired)
					N_resp_paired = len(resp_paired)


					if N_stim_paired == N_resp_paired:

						# Calculate and save asynchronies.
						for k in range(N_stim_paired):
							diff = resp_paired[k]-stim_paired[k]
							if abs(diff)<200:
								asynchrony.append(diff)
							else:
								print('Error tipo OOT')
								errors.append('OutOfThreshold')
								raise Error

						# If the code got here, then the trial is valid!:
						valid_trials.append(1)
						errors.append('NoError') 
                        
					#==============================================================================
					# Plot all pair of stimulus and feedback
						my_labels = {"stim" : "Stimulus", "resp" : "Response"}
						for j in range(N_stim):
							plt.axvline(x=stim_time[j],color='b',linestyle='dashed',label=my_labels["stim"])
							my_labels["stim"] = "_nolegend_"

						for k in range(N_resp):
							plt.axvline(x=resp_time[k],color='r',label=my_labels["resp"])
							my_labels["resp"] = "_nolegend_"

						# Put a yellow star on the stimulus that have a paired response.
						for j in range(N_stim_paired):
							plt.plot(stim_paired[j],0.5,'*',color='y')

						plt.axis([min(stim_time)-50,max(resp_time)+50,0,1])

						plt.xlabel('Tiempo[ms]',fontsize=12)
						plt.ylabel(' ')
						plt.grid()    
						plt.legend(fontsize=12)
					#==============================================================================

						# Go to next trial.
						trial = trial + 1

					else:
						if N_stim_paired > N_resp_paired: # If subject skipped an stimul.
							# Trial is not valid! then:
							print('Error tipo SS')
							errors.append('SkipStim')
						else: # If there's too many responses.
							# Trial is not valid! then:
							print('Error tipo TMR')
							errors.append('TooManyResp')
							raise Error


				else: # If there were no responses.
					# Trial is not valid! then:
					print('Error tipo NR')
					errors.append('NoResp')  
					raise Error


			except (Error):
				# Trial is not valid! then:
				valid_trials.append(0)


				# Add 1 to number of trials per block since will have to repeat one.
				block_conditions_df = block_conditions_df.append(block_conditions_df.iloc[trial]).reset_index()
				block_conditions_df = block_conditions_df.drop(columns = ['Trial'])
				block_conditions_df.index.name="Trial"

				# Go to next trial.
				trial = trial + 1

			# SAVE DATA FROM TRIAL (VALID OR NOT).
			f_data_dict = {'Data' : data, 'Stim_time' : stim_time, 'Resp_time' : resp_time, 'Asynchrony' : asynchrony}   
			f_data_str = json.dumps(f_data_dict)
			f_data = open(filename_data, "w")
			f_data.write(f_data_str)
			f_data.close()

	#==============================================================================
	#         # If you want to show plots for each trial.
	#         plt.show(block=False)
	#         plt.show()
	#         plt.pause(0.5)
	#==============================================================================

		print("Fin del bloque!\n")


		# SAVE DATA FROM BLOCK (VALID AND INVALID TRIALS, MESSAGES AND ERRORS).    
		block_conditions_df = block_conditions_df.assign(Valid_trial = valid_trials, Message = messages, Error = errors)
		block_conditions_df.insert(1, 'Subject', curr_subject_number_int)
		block_conditions_df.to_csv(filename_block)

		# Go to next block.
		block_counter = block_counter + 1
    
	print("Fin del experimento!")
	arduino.close()

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


# This function plots the asynchronys of a trial. Must be called from the console: Plot_asynch(asynchrony).
def Plot_asynch(asynch_vector):
	plt.figure(2)
	plt.plot(asynch_vector,'.-')
	plt.xlabel('# beep',fontsize=12)
	plt.ylabel('Asynchrony[ms]',fontsize=12)
	plt.grid() 


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
def Load_ExpMetadata():
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


#%% Load_TrialData
# Function to load trial data from trial data file. Return a dataframe.
# subject_number --> int (ej: 0). block --> int (ej: 1). trial --> int (ej: 2).
def Load_TrialData(subject_number, block, trial):
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
		subject_n.append(s_number)
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
			
	asyn_df = Compute_Asyn(stim_times,resp_times)
	for i in range(len(asyn_df.index)):
		event.append('A')
		time.append(asyn_df['asyn'][i])
		subject_n.append(s_number)
		block_n.append(block)
		trial_n.append(trial)
		order.append(asyn_df['assigned_stim'][i])
	
	trialData_df = pd.DataFrame()
	trialData_df = trialData_df.assign(Subject = subject_n, Block = block_n, Trial = trial_n, Event = event, Event_Order = order, Time = time)
	trialData_df.to_csv('./Data/TrialData.csv')
	return trialData_df


#%% Load_TrialsData
# Function to load trials data. Return a dataframe.
def Load_TrialsData():
	expMetadata_df = Load_ExpMetadata()
	conc_trialData_df = pd.DataFrame()
	for i in range(len(expMetadata_df.index)): 
		subject_number = expMetadata_df['Subject'][i]
		block = expMetadata_df['Block'][i]
		trial = expMetadata_df['Trial'][i]
		trialData_df = Load_TrialData(subject_number, block, trial)
		conc_trialData_df = conc_trialData_df.append(trialData_df)
	conc_trialData_df = conc_trialData_df.reset_index()
	conc_trialData_df = conc_trialData_df.drop(columns = ['index'])
	conc_trialData_df.to_csv('./Data/TrialsData.csv')
	return conc_trialData_df


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
	
		# keep track of which stimulus was assigned to each response (NaN if not assigned)
		assigned_stim[R_idx] = S_idx
		# keep track of assigned stimuli (-1 if not assigned)
		stimuli_flag[S_idx] = np.ones(len(S_idx))
	
		# save asynchrony (NA means that response was not assigned)
		if (S_idx.size != 0 and R_idx.size != 0):
			lin_ind = np.ravel_multi_index((S_idx,R_idx),differences.shape)
			asyn[R_idx] = differences.ravel()[lin_ind]
	
	# Output
	output = pd.DataFrame({'asyn':asyn,'assigned_stim':assigned_stim})
	return output

