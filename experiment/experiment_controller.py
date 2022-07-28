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
import pandas as pd
import json
import tappinduino as tp


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
perturb_type = [1,2]     	 	# Perturbation type. 0--> No perturbation. 1--> Step change. 2--> Phase shift.
perturb_size = 100           	# Perturbation size.
perturb_bip_range = (5,7)		# Perturbation bip range.
max_stim_for_first_resp = 5		# Maximum number of stimulus for first response


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
			curr_subject_number_int = last_subject_number
			curr_subject_number = '{0:0>3}'.format(curr_subject_number_int)
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

			# Close raw data file.    
			f_raw.close()

			# ---------------------------------------------------------------
			# Asynchronies calculation.

			# Vector that will contain asynchronies if they are calculated.
			asyn_df = tp.Compute_Asyn(stim_time,resp_time)
			error_handling = tp.Error_Handling(asyn_df, resp_time, max_stim_for_first_resp)
			errors.append(error_handling[1])
			valid_trials.append(error_handling[2])
			if (error_handling[2] == 0):
				print(error_handling[0])
				# Add 1 to number of trials per block since will have to repeat one.
				block_conditions_df = block_conditions_df.append(block_conditions_df.iloc[trial]).reset_index()
				block_conditions_df = block_conditions_df.drop(columns = ['Trial'])
				block_conditions_df.index.name="Trial"
 
			# SAVE DATA FROM TRIAL (VALID OR NOT).
			f_data_dict = {'Data' : data, 'Stim_time' : stim_time, 'Resp_time' : resp_time, 'Asynchrony' : asyn_df['asyn'].tolist(), 'Stim_assigned_to_asyn' : asyn_df['assigned_stim'].tolist()}   
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

			# Go to next trial.
			trial = trial + 1

		
		print("Fin del bloque!\n")


		# SAVE DATA FROM BLOCK (VALID AND INVALID TRIALS, MESSAGES AND ERRORS).    
		block_conditions_df = block_conditions_df.assign(Valid_trial = valid_trials, Message = messages, Error = errors)
		block_conditions_df.insert(1, 'Subject', curr_subject_number_int)
		block_conditions_df.to_csv(filename_block)

		# Go to next block.
		block_counter = block_counter + 1
    
	print("Fin del experimento!")
	arduino.close()


#%% Load_TrialsData
# Function to load trials data. Return a dataframe.
tp.Load_TrialsData(n_blocks)


#%% Plot_Trials_Asyn_PerSubject_PerCondition.
# Function to plot all trials asynchronies per subject and per condition.
# subject --> int (ej: 0). condition --> int (ej: 1). figure_number --> int (ej: 1).
tp.Plot_Trials_Asyn_PerSubject_PerCondition(0, 0, 1)


#%% Plot_Mean_Trials_Asyn_AllSubjects_PerCondition.
# Function to plot mean trials asynchronies for all subjects and per condition.
# condition --> int (ej: 1). figure_number --> int (ej: 1).
tp.Plot_Mean_Trials_Asyn_AllSubjects_PerCondition(0, 2)


#%% Plot_Mean_Trials_Asyn_AllSubjects_AllConditions
# Function to plot mean trials asynchronies across all subjects and for all conditions.
# n_conditions --> int (ej: 3). figure_number --> int (ej: 1).
tp.Plot_Mean_Trials_Asyn_AllSubjects_AllConditions(n_conditions, 3)
		
	