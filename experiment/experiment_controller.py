# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:04:30 2019

@author: paulac
"""

import serial, time
import numpy as np
import numpy.matlib as mlib
import matplotlib.pyplot as plt
import random
import os
from itertools import permutations 
import glob
import pickle
import pandas as pd
import csv


#%% Description

#==============================================================================
# Saves:  - a file per block containing information about all trials in it: their condition and whether if they were valid or not
#         - a file per trial containing the raw data from it
#         - a file per trial containing extracted data from it
#==============================================================================

#%% Communicate with arduino

arduino = serial.Serial('COM4', 9600)


#%% Definitions

# Define variables

ISI = 500		             # Interstimulus interval (milliseconds)
n_stim = 35	                 # Number of bips within a sequence
n_trials_perblock = 20       # Number of trials per conditions
n_blocks = 3                 # Number of blocks
n_subj_max = 100             # Maximum number of subjects
perturb_type = 2             # Perturbation type
perturb_size = 100           # Perturbation size


#%% Conditions

# Filename for the file that will contain all possible permutations for the subjects
presentation_orders = './Data/Presentation_orders.csv'

# Number of trials
n_trials = n_trials_perblock * n_blocks

# All possible conditions for perturbations
all_conditions = ['pos', 'neg', 'iso']
n_conditions = len(all_conditions)      # number of conditions

# Condition dictionary so we can choose the condition without going through number position
condition_dictionary = {"pos": 0,"neg": 1,"iso": 2}

# Start experiment or generate Perturbation_orders.csv file
start_or_generate_response = input("Presione enter para iniciar experimento, o escriba la letra G (generar archivo con órdenes de presentación) y presione enter: ") 

# If this is the first time running the experiment, then it's necessary to generate the Presentation_orders.csv file
if start_or_generate_response == 'G':
    
    confirm_response = input('¿Está seguro? Si el archivo ya existe se sobrescribirá. Escriba S y presione enter para aceptar, o sólo presione enter para cancelar: ')
    
    if confirm_response == 'S':
        chosen_conditions = mlib.repmat(np.arange(0,n_conditions),n_subj_max,n_trials_perblock)
        for i in range(0,n_subj_max):
            random.shuffle(chosen_conditions[i])
       
        presentation_orders_df = pd.DataFrame()
                      
        for i in range(0,n_subj_max):
            next_subject_number = '{0:0>3}'.format(i)
            next_subject_number = 'S' + next_subject_number
            presentation_orders_df[next_subject_number] = chosen_conditions[i]
        
        presentation_orders_df.index.name="Trial"
  
        presentation_orders_df.to_csv(presentation_orders)
        
    
#%%      
    
    
    # vector that will contain all conditions needed for this experiment    
    #inputs = []
    #inp = input("Elija las condiciones que quiere usar separandolas por Enters. Al finalizar vuelva a presionar Enter.\n")
    #while inp != "":
    #    inputs.append(inp)  
    #    inp = input()
    #    if inp == "":
    #        break
        
    # finding the conditions index in the dictionary
    #chosen_conditions = []    
    #for condition in inputs:
    #    chosen_conditions.append(condition_dictionary[condition])
        
    # find all possible permutations of these conditions    
    #all_possible_orders_conditions = list(permutations(chosen_conditions))
    # save them in a file
    #with open(presentation_orders, 'wb') as fp:
    #    pickle.dump(all_possible_orders_conditions, fp)

# if this isn't the first time running the experiment, the file with all possible permutations should already exist and have the information of the permutations already used by subjects (those permutations will no longer be in the file)
#else:
    # try to find the file. If it doesn't exist in the directory, raise an error.
#    try:
#        f_orders = open(presentation_orders,"r")
#        pass
        
#    except IOError:
#        print('El archivo con las permutaciones de las condiciones no esta donde deberia, ubicalo en la carpeta correcta y volve a correr esta celda')
#        raise
    
#%% Definitions

# Define variables

#ISI = 500;		# interstimulus interval (milliseconds)
#n_stim =30;	# number of bips within a sequence

#with open (presentation_orders, 'rb') as fp:
#    content = pickle.load(fp)
# total number of blocks (equal to number of conditions since we have one condition per block)
#N_blocks = len(content[0]);
# number of trials per condition per block
#N_trials_per_block_per_cond = 2;

# Open Presentation_orders.csv file as dataframe
presentation_orders_df = pd.read_csv(presentation_orders,index_col='Trial')

# Define Python user-defined exceptions
class Error(Exception):
   """Base class for other exceptions"""
   pass

#%% Experiment

# Check for file with names and pseudonyms
filename_names = './Data/Dic_names_pseud.dat'

try:
    f_names = open(filename_names,"r")

    if os.stat(filename_names).st_size == 0:
        curr_subject_number = '001';
        f_names.close();
    else:
        content = f_names.read();
        last_subject_number = int(content [-3:]);
        curr_subject_number = '{0:0>3}'.format(last_subject_number + 1);
        f_names.close()
        
except IOError:
    print('El archivo no esta donde deberia, ubicalo en la carpeta correcta y volve a correr esta celda')
    raise

# set subject name for filename
name = input("Ingrese su nombre: ") 

f_names = open(filename_names,"a")
f_names.write('\n'+name+'\tS'+curr_subject_number)
f_names.close()

#with open (presentation_orders, 'rb') as fp:
#    content = pickle.load(fp)
#    print(fp)
    
#cond_order_block = random.choice(content)

#content.pop(content.index(cond_order_block))

#with open(presentation_orders, 'wb') as fp:
#    pickle.dump(content, fp)

# Run blocks
block_counter = 0

# Trials for the current subject
subject_df = pd.DataFrame(presentation_orders_df['S' + curr_subject_number])


while (block_counter < n_blocks):
    
    # Block conditions
    block_conditions_aux = block_counter * n_trials_perblock
    block_conditions_df = subject_df.loc[block_conditions_aux : block_conditions_aux + n_trials_perblock - 1]
    perturb_bip_list = []
    perturb_size_list = []
    for i in range(0,n_trials_perblock):
        perturb_bip_list.append(random.randrange(10,15,1))
        trial_type = (block_conditions_df.loc[[block_conditions_aux + i]].values.tolist())[0][0]
        if (trial_type == 0):
            perturb_size_list.append(perturb_size)        # Perturbation size 
        elif (trial_type == 1):
            perturb_size_list.append(-perturb_size)       # Perturbation size 
        else:
            perturb_size_list.append(0)                   # Perturbation size 
    block_conditions_df = block_conditions_df.assign(Perturb_bip = perturb_bip_list, Perturb_size = perturb_size_list, Original_trial = range(block_conditions_aux,block_conditions_aux + n_trials_perblock))
    
    #condition_vector = [] # vector that will contain the specified condition the correct amount of times (it's important to restart it here!)
    #for i in range(N_trials_per_block_per_cond):
    #    condition_vector.append(all_conditions[cond_order_block[block_counter]])
    # total number of trials per block
    #n_trials_perblock = len(condition_vector) # unlike N_trials_per_block_per_cond this variable will change if a trial goes wrong

    #Stim_conds = [] # vector that will contain all stimulus conditions
    #Fdbk_conds = [] # vector that will contain all feedback conditions
    #for i in range(len(condition_vector)):
    #    Stim_conds.append(condition_vector[i][0])
    #    Fdbk_conds.append(condition_vector[i][1])
    
    # Run one block
    input("Presione Enter para comenzar el bloque (%d/%d):" %  (block_counter+1,n_blocks));
    
    # Set time for file name
    timestr = time.strftime("%Y_%m_%d-%H.%M.%S")
    
    trial = 0
    
    conditions = [] # vector that will contain exact message sent to arduino to register the conditions played in each trial
    valid_trials = [] # vector that will contain 1 if the trial was valid or 0 if it wasn't
    errors = [] # vector that will contain the type of error that ocurred if any did    
    
    # Generate filename for file that will contain all conditions used in the trial along with the valid_trials vector    
    filename_block = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-trials"
    
    while (trial < len(block_conditions_df.index)):
        input("Presione Enter para comenzar el trial (%d/%d):" % (trial+1,len(block_conditions_df.index)));
        plt.close(1)
        plt.close(2)
        
        # Generate raw data file 
        filename_raw = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)+"-raw.dat"
        f_raw = open(filename_raw,"w+")
     
        # Generate extracted data file name (will save raw data, stimulus time, feedback time and asynchrony)
        filename_data = './Data/S'+curr_subject_number+"-"+timestr+"-"+"block"+str(block_counter)+"-"+"trial"+str(trial)    
        
        # Wait random number of seconds before actually starting the trial
        wait = random.randrange(10,20,1)/10.0
        time.sleep(wait)
        
        # Define stimulus and feedback condition for this trial
        #Stim = Stim_conds[trial];
        #Resp = Fdbk_conds[trial];
        perturb_size_aux = (block_conditions_df.loc[[trial],['Perturb_size']].values.tolist())[0][0]
        perturb_bip_aux = (block_conditions_df.loc[[trial],['Perturb_bip']].values.tolist())[0][0]
                  
        # Send message with conditions to arduino
        message = str.encode(";S%c;F%c;N%c;A%d;I%d;n%d;P%d;B%d;T%d;X" % ('B', 'B', 'B', 3, ISI, n_stim, perturb_size_aux, perturb_bip_aux, perturb_type))
        arduino.write(message)
        conditions.append(message)
        #time.sleep(25)            

        # Read information from arduino
        data = []
        aux = arduino.readline().decode()
        while (aux[0]!='E'):
            data.append(aux)
            f_raw.write(aux) # save raw data
            aux = arduino.readline().decode()
                   
        # Separates data in type, number and time
        e_total = len(data)
        e_type = []
        e_number = []
        e_time = []
        for event in data:
            e_type.append(event.split()[0])
            e_number.append(int(event.split()[1]))
            e_time.append(int(event.split()[2]))
        
        # Separates number and time according to if it comes from stimulus or response
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
    
        # determine number of stimulus and responses registered
        N_stim = len(stim_time)
        N_resp = len(resp_time)
    
        # close raw data file    
        f_raw.close()
        
        # ---------------------------------------------------------------
        # Asynchronies calculation
    
        # vector that will contain asynchronies if they are calculated
        asynchrony = []
        
        try: 
            if N_resp > 0: # if there were any responses
            
                j = 0; # stimulus counter
                k = 0; # responses counter for finding first stimuli with decent response
                i = N_resp-1; # responses counter for finding last stimuli with response
                first_stim_responded_flag = False; # flag if there was a stimuli with a recent response
                last_resp_flag = False;                
                
                
                # find first stimulus with a decent response
                while j < 5: # if the first response doesn't match with any of the 5 first stimuli, then re-do the trial
                    diff = stim_time[j]-resp_time[k];
                    if abs(diff)<200:
                        first_stim_responded_index = j;
                        first_stim_responded_flag = True;
                        break;
                    else:
                        j = j+1;

                
                if first_stim_responded_flag == True:
                    pass;
                else:
                    print('Error tipo NFR')
                    errors.append('NoFirstResp')
                    raise Error 
                                
                
                # find response to last stimulus (last response that should be considerated)
                while i > 0:
                    diff = stim_time[N_stim-1]-resp_time[i]
                    if abs(diff)<200:
                        last_resp_index = i;
                        last_resp_flag = True;
                        break;
                    else:
                        i = i-1;
                        
                if last_resp_flag == True:
                    pass;
                else:
                    print('Error tipo NLR')
                    errors.append('NoLastResp')
                    raise Error 
                            
                
                # new vectors of stimulus and responses that only contain those that have a pair of the other type        
                stim_paired = stim_time[first_stim_responded_index:N_stim]
                resp_paired = resp_time[0:(last_resp_index+1)]
                N_stim_paired = len(stim_paired)
                N_resp_paired = len(resp_paired)
                
                if N_stim_paired == N_resp_paired:
                                      
                    
                    # Calculate and save asynchronies
                    for k in range(N_stim_paired):
                        diff = resp_paired[k]-stim_paired[k]
                        if abs(diff)<200:
                            asynchrony.append(diff)
                        else:
                            print('Error tipo OOT')
                            errors.append('OutOfThreshold')
                            raise Error
                            
                             
                    # if the code got here, then the trial is valid!:
                    valid_trials.append(1)
                    errors.append('NoError') 
                    
                #==============================================================================
                # Plot all pair of stimulus and feedback
#                    plt.figure(1)
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
           
                #==============================================================================
                # Plot asynchronies
#                    plt.figure(2)
#                    plt.plot(asynchrony,'.-')
#                    plt.xlabel('# beep',fontsize=12)
#                    plt.ylabel('Asynchrony[ms]',fontsize=12)
#                    plt.grid()    
                #==============================================================================
            
                    # go to next trial
                    trial = trial + 1;
                
                else:
                    if N_stim_paired > N_resp_paired: # if subject skipped an stimuli
                        # trial is not valid! then:
                        print('Error tipo SS')
                        errors.append('SkipStim')
                    else: # if there's too many responses
                        # trial is not valid! then:
                        print('Error tipo TMR')
                        errors.append('TooManyResp')
                    
                    raise Error
                    
                      
            else: # if there were no responses
                # trial is not valid! then:
                print('Error tipo NR')
                errors.append('NoResp')  
                raise Error
             
               
        except (Error):
            # trial is not valid! then:
            valid_trials.append(0)
                
            # appends conditions for this trial at the end of the conditions vectors, so that it can repeat at the end
            #Stim_conds.append(Stim_conds[trial])
            #Fdbk_conds.append(Fdbk_conds[trial])
                      
            # Add 1 to number of trials per block since will have to repeat one
            #n_trials_perblock = n_trials_perblock + 1;
            block_conditions_df = block_conditions_df.append(block_conditions_df.iloc[trial]).reset_index()
            block_conditions_df.index = block_conditions_df.index + block_conditions_aux
            block_conditions_df = block_conditions_df.drop(columns = ['Trial'])
            block_conditions_df.index.name="Trial"
            
            # Go to next trial
            trial = trial + 1;
            
            

        # SAVE DATA FROM TRIAL (VALID OR NOT)
        np.savez_compressed(filename_data, raw=data, stim=stim_time, resp=resp_time, asynch=asynchrony)
        
        data_trial = {'Data' : data, 'Stim_time' : stim_time, 'Resp_time' : resp_time, 'Asynchrony' : asynchrony}   
      

#==============================================================================
#         # If you want to show plots for each trial
#         plt.show(block=False)
#         plt.show()
#         plt.pause(0.5)
#               
#==============================================================================

    print("Fin del bloque!")

    # ask subject what condition of stimulus and responses considers he/she heard
    #stim_subject_percep = input("Considera que el estimulo llegó por audio izquierdo(L), derecho(R) o ambos(B)?") 
    #fdbk_subject_percep = input("Considera que su respuesta llegó por audio izquierdo(L), derecho(R) o ambos(B)?") 
    #block_cond_subject_percep = [stim_subject_percep, fdbk_subject_percep]
    
    # SAVE DATA FROM BLOCK (VALID AND INVALID TRIALS AND THEIR CONDITIONS)    
    np.savez_compressed(filename_block,trials=valid_trials,conditions=conditions,errors=errors)
    
    # go to next block
    block_counter = block_counter +1;

print("Fin del experimento!")

#%% A look at the last trial


def Plot_RespStim(stim_vector,resp_vector):
    N_stim = len(stim_vector)
    N_resp = len(resp_vector)
    #plt.figure()
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

 #Put a yellow star on the stimulus that have a paired response.
#for j in range(N_stim_paired):
#    plt.plot(stim_paired[j],0.5,'*',color='y')
#    
    
#my_labels = {"stim-paired" : "Stimulus-paired", "resp-paired" : "Response-paired"}
#for j in range(N_stim_paired):
#    plt.axvline(x=stim_paired[j],color='y',linestyle='dashed',label=my_labels["stim-paired"])
#    my_labels["stim-paired"] = "_nolegend_"
##
#for k in range(N_resp_paired):
#    plt.axvline(x=resp_paired[k],color='c',label=my_labels["resp-paired"])
#    my_labels["resp-paired"] = "_nolegend_"    
#    

def Plot_asynch(asynch_vector):
    plt.figure(2)
    plt.plot(asynch_vector,'.-')
    plt.xlabel('# beep',fontsize=12)
    plt.ylabel('Asynchrony[ms]',fontsize=12)
    plt.grid() 


#%% Loading data

# Function for loading data specific data from either the block or trial files.
def Loading_data(subject_number,block, trial, *asked_data):
    # IMPORTANTE: DAR INPUTS COMO STRING
    # Hay que darle si o si numero de sujeto y bloque y el trial puede estar especificado o ser None. Recordar que los archivos que no tienen identificado el trial tienen guardada la informacion de todo el bloque: condicion usada, errores, percepcion del sujeto y si el trial fue valido o invalido. En cambio, al especificar el trial se tiene la informacion de cada trial particular, es decir, asincronias, datos crudos, respuestas y estimulos.

    if trial is None:
        #file_to_load = glob.glob(r'C:\Users\Paula\Documents\Facultad\Tesis de licenciatura\tappingduino 3\Codigo\2020\DATOS/S'+subject_number+"*-block"+str(block)+"-trials.npz")
        # file_to_load = glob.glob('/home/paula/Tappingduino3/tappingduino-3-master/Datos/S'+subject_number+"*-block"+str(block)+"-trials.npz")
        file_to_load = glob.glob(r'C:\\Users\\Administrator\\Documents\\Ariel\\DOCTORADO\\DOCTORADO TRABAJANDO\\Python\\Datos experimento\\experimento\\Data\\S'+subject_number+"*-block"+str(block)+"-trials.npz")
    else:
        #file_to_load = glob.glob(r'C:\Users\Paula\Documents\Facultad\Tesis de licenciatura\tappingduino 3\Codigo\2020\DATOS/S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz")    
        file_to_load = glob.glob(r'C:\\Users\\Administrator\\Documents\\Ariel\\DOCTORADO\\DOCTORADO TRABAJANDO\\Python\\Datos experimento\\experimento\\Data\\S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz")
    
        #file_to_load = glob.glob('/home/paula/Tappingduino3/tappingduino-3-master/Datos/S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz")    
    
    npz = np.load(file_to_load[0])
    if len(asked_data) == 0:
        print("The file contains:")
        return sorted(npz)
    else:
        data_to_return = []
        for a in asked_data:
            data_to_return.append(npz[a])                                
        return data_to_return[:]


#%% Testing Loading_data and plotting asynchronies

asynch = Loading_data('003',3,3,'asynch')
plt.plot(asynch[0],'.-')
plt.xlabel('# beep',fontsize=12)
plt.ylabel('Asynchrony[ms]',fontsize=12)
plt.grid() 


#%% Load asynchronies

# Loads all asynchronies for a subject for an specific block and returns all plots
def Loading_asynch(subject_number,block):
    file_to_load = glob.glob(r'C:\\Users\\Administrator\\Documents\\Ariel\\DOCTORADO\\DOCTORADO TRABAJANDO\\Python\\Datos experimento\\experimento\\Data\\S'+subject_number+"*-block"+str(block)+"-trials.npz")
    #file_to_load = glob.glob('/home/paula/Tappingduino3/tappingduino-3-master/Datos/S'+subject_number+"*-block"+str(block)+"-trials.npz")    
    npz = np.load(file_to_load[0])
    trials = npz['trials']
    
    valid_index = []
    for i in range(len(trials)):
        if trials[i] == 1:
            valid_index.append(i)
    
    for trial in valid_index:
        #file_to_load_trial = glob.glob(r'C:\Users\Paula\Documents\Facultad\Tesis de licenciatura\tappingduino 3\Codigo\2020\DATOS/S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz")    
        # file_to_load_trial = glob.glob('/home/paula/Tappingduino3/tappingduino-3-master/Datos/S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz")    
        file_to_load_trial = glob.glob(r'C:\\Users\\Administrator\\Documents\\Ariel\\DOCTORADO\\DOCTORADO TRABAJANDO\\Python\\Datos experimento\\experimento\\Data\\S'+subject_number+"*-block"+str(block)+"-trial"+str(trial)+".npz") 
        npz_trial = np.load(file_to_load_trial[0])
        asynch_trial = npz_trial['asynch']
        plt.plot(asynch_trial,'.-', label = 'trial %d' % trial)
    plt.xlabel('# beep',fontsize=12)
    plt.ylabel('Asynchrony[ms]',fontsize=12)
    plt.grid()  
    plt.legend()

    return

