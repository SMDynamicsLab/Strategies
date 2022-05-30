# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:13:56 2020

@author: Paula

Este archivo genera todos los gráficos y datos utilizados en el análisis de
datos de la tesis, tomando la información del archivo que devuelve
"LimpiezaDatos.py"

IMPORTANTE:
Para que el código funcione correctamente, el archivo .pkl que devuelve
"LimpiezaDatos.py" debe estar en el directorio donde se está trabajando.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
from scipy.stats import ttest_rel

# %% Cargo el archivo que salió de LimpiezaDatos.py
data = pd.read_pickle("valid_subjects_data_PICKL.pkl")

# %% DEFINO FUNCIONES y PALETA DE COLORES
# Paleta Colorblind-friendly (repito dos veces los 8 colores para que no se
# quede sin colores al graficar los trials..)

cbPaleta = ["#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
            "#D55E00", "#CC79A7", "#999999", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]


def pos_estim(cond):
    '''
    Devuelve L o R dependiendo de si en la condición que está evaluando el
    estímulo está en izquierda o en derecha, respectivamente.
    Lo uso para tener la información para hacer el test de ANOVA.

    Parameters
    ----------
    cond: String con la condición a estudiar.

    Returns
    -------
    String L o R según corresponda.

    '''
    if cond == 'LL' or cond == 'LR':
        return 'L'
    else:
        return 'R'


def par_stimfdbk(cond):
    '''
    Devuelve "same" o "diff" ("mismo" o "diferente") dependiendo de si en la
    condición que está evaluando el estímulo y el feedback están ambos en el
    mismo oído o en diferentes.
    Lo uso para tener la información para hacer el test de ANOVA.

    Parameters
    ----------
    cond: String con la condición a estudiar.

    Returns
    -------
    String "same" o "diff" según corresponda.

    '''
    if cond == 'LL' or cond == 'RR':
        return 'same'
    else:
        return 'diff'


# %% PLOT FINAL i: asyn vs bip single trial de un sujeto muestra (sujeto 1 en
# este caso) en BB.

asynch = data['BB']['sujeto 1'].loc['trial11']

asynch.plot(use_index=False, style='.-', figsize=(10, 8))
plt.xlabel('# bip', fontsize=18)
plt.ylabel('$e_n$[ms]', fontsize=18)
plt.grid(True)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Asincronia en cada estimulo para el sujeto 1 en la condicion BB')
# plt.savefig('SingleTrial_esquema.png')

# %% PLOT FINAL iii: Todos los trials de un sujeto en BB (Gráfico tipo 1).
subject = 'sujeto 1'
condition = 'BB'

# Es necesario trasponer el dataframe (.T) para que los beeps queden en
# el eje X
data[condition][subject].T.plot(use_index=False,
                                style='.-',
                                color=cbPaleta,
                                figsize=(10, 8))
plt.xlabel('# beep', fontsize=15)
plt.ylabel('Asincronia[ms]', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend()
plt.title('Todos los trials validos del %s en la condicion %s'
          % (subject, condition), fontsize=15)
# plt.savefig('Asynch_%s_%s_alltrials.png' % (subject, condition))


# AGREGO DISTRIBUCIÓN DE ASINCRONÍAS
plt.figure()
data['BB']['sujeto 1'].T.stack().plot.hist(bins=30,
                                           figsize=(10, 8),
                                           color=cbPaleta[5])
plt.grid()
plt.xlabel('Asincronia [ms]', fontsize=18)
plt.ylabel('')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.savefig('Distrib_Asynch_%s_%s.png' % (subject, condition))

# %% PLOT FINAL iv: Todos los sujetos un gráfico por cada condición.

condition_vector = ['LL', 'RR', 'BB', 'RL', 'LR']
total_number_subjects = len(data)

# Los voy a poner a los 5 gráficos en una sola figura como subplots
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

for condition in condition_vector:
    # Quiero a cada condición en una ubicación específica en la figura acorde
    # a como presentamos las condiciones de ANOVA. Esto es prescindible si
    # sólo se quieren los gráficos
    if condition == 'LL':
        j = 1
    if condition == 'LR':
        j = 2
    if condition == 'RR':
        j = 3
    if condition == 'RL':
        j = 4
    if condition == 'BB':
        j = 5
    ax = fig.add_subplot(3, 2, j)

    for subject in range(1, total_number_subjects+1):
        trial = data[condition]['sujeto %i' % subject]
        trial_promedio = trial.mean()
        trial_promedio_err = trial.std()/np.sqrt(len(trial_promedio))
        trial_promedio.plot(use_index=False,
                            label='Sujeto %i' % subject,
                            style='.-',
                            color=cbPaleta[subject+4],
                            yerr=trial_promedio_err)

    plt.axis([-2, 42, -55, 15])
    plt.xlabel('# beep', fontsize=15)
    plt.ylabel('Asincronia[ms]', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.title('Condicion %s' % condition, fontsize=15)

fig.tight_layout()
# fig.savefig('mean_across_trials_ALLconditions.png')

# %% PLOT FINAL v y vi: Gráfico de asyn media, 5 condiciones en el eje x:
# una línea para el valor medio total con incertezas + 1 línea por cada sujeto
# sin incerteza (Gráfico 5 + Gráfico 7 sin incertezas).

# Creo un DataFrame vacío para llenar con los valores promedio.
# Elijo hacerlo así y no directamente al plotear porque para calcular los
# promedios entre sujetos para cada condición, necesito volver a calcular todo
# nuevamente. Usando el dataframe, solo calculo las cosas una vez.
meandata = pd.DataFrame([],
                        columns=condition_vector,
                        index=['sujeto %i'
                               % i for i in range(1, total_number_subjects+1)])
# Relleno el DataFrame
for subject in range(1, total_number_subjects+1):
    for condition in condition_vector:
        trial = data[condition]['sujeto %i' % subject]
        meandata[condition]['sujeto %i' % subject] = trial.mean(axis=1).mean()

# Hago el plot (notar que el plot de BB está aparte porque así decidimos
# mostrar la información)
data_sinBB = meandata.loc[:, meandata.columns != 'BB']
plt.figure()
data_sinBB.T.plot(figsize=(10, 8),
                  style='.-',
                  linewidth=4,
                  markersize=15,
                  color=cbPaleta[5:])

# Agrego el promedio entre sujetos
data_sinBB.mean().plot(fmt='X-',
                       capsize=10,
                       markersize=10,
                       color='k',
                       label='Promedio de todos los sujetos',
                       yerr=data_sinBB.std()/np.sqrt(len(data_sinBB)))

# Para la parte de BB: uso un for porque quiero colores diferentes
# para cada punto (ya que cada punto representa un sujeto)
for i in range(total_number_subjects):
    plt.plot(4, meandata['BB'][i], '.', color=cbPaleta[i+5], markersize=15)

plt.errorbar(4, meandata['BB'].mean(),
             fmt='X-',
             capsize=10,
             markersize=10,
             color='k',
             yerr=meandata['BB'].std()/np.sqrt(len(meandata['BB'])))

# Burocracias y embellecimiento del gráfico
condition_x = [0, 1, 2, 3, 4]
condition_labels = ['LL', 'RR', 'RL', 'LR', 'BB']
plt.xticks(condition_x, condition_labels, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Asincronia promedio [ms]', fontsize=15)
plt.xlabel('Condiciones', fontsize=15)
plt.grid(True)
plt.legend()
plt.title('Promedio along y across trials para cada sujeto en cada condicion')
# plt.savefig('Mean_Unicovalor_xSubjxCond.png')

# Hago exactamente lo mismo pero ahora con los valores de std media
# (Gráfico 6 + Gráfico 8 sin incertezas)
stddata = pd.DataFrame([],
                       columns=condition_vector,
                       index=['sujeto %i'
                              % i for i in range(1, total_number_subjects+1)])
# Relleno el DataFrame
for subject in range(1, total_number_subjects+1):
    for condition in condition_vector:
        trial = data[condition]['sujeto %i' % subject]
        stddata[condition]['sujeto %i' % subject] = (trial.std(axis=1) /
                                                     np.sqrt(len(trial.columns)
                                                             )).mean()

# Hago el plot (notar que el plot de BB está aparte porque así decidimos
# mostrar la información)
stddata_sinBB = stddata.loc[:, stddata.columns != 'BB']
plt.figure()
stddata_sinBB.T.plot(figsize=(10, 8),
                     style='.-',
                     linewidth=4,
                     markersize=15,
                     color=cbPaleta[5:])

# Agrego el promedio entre sujetos
stddata_sinBB.mean().plot(fmt='X-',
                          capsize=10,
                          markersize=10,
                          color='k',
                          label='Promedio de todos los sujetos',
                          yerr=stddata_sinBB.std()/np.sqrt(len(stddata_sinBB)))

# Para la parte de BB: uso un for porque quiero colores diferentes
# para cada punto (ya que cada punto representa un sujeto)
for i in range(total_number_subjects):
    plt.plot(4, stddata['BB'][i], '.', color=cbPaleta[i+5], markersize=15)

plt.errorbar(4, stddata['BB'].mean(),
             fmt='X-',
             capsize=10,
             markersize=10,
             color='k',
             yerr=stddata['BB'].std()/np.sqrt(len(stddata['BB'])))

# Burocracias y embellecimiento del gráfico
condition_x = [0, 1, 2, 3, 4]
condition_labels = ['LL', 'RR', 'RL', 'LR', 'BB']
plt.xticks(condition_x, condition_labels, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Desviación estándar promedio [ms]', fontsize=15)
plt.xlabel('Condiciones', fontsize=15)
plt.grid(True)
plt.legend()
plt.title('Promedio along y across trials para cada sujeto en cada condicion')
# plt.savefig('Std_Unicovalor_xSubjxCond.png')

# %%
# Plot final vii y viii: Repeated measures two-way anova para asyn media y std
# media (Gráfico 5.1 y 6.1 + Tablas)

# Primero necesito reacomodar el DataFrame para tener todo como columnas y
# poder dárselo a la funcion de repeated measures de ANOVA.
data_anova = pd.melt(meandata.loc[:, meandata.columns != 'BB'].reset_index(),
                     id_vars='index',
                     var_name=['condition'])

# Agrego una columna con la información del factor "Par estim-fdbk"
data_anova['par_sf'] = data_anova.apply(lambda row:
                                        par_stimfdbk(row.condition),
                                        axis=1)
# Agrego una columan con la información del factor "Posicion del estímulo"
data_anova['pos_estim'] = data_anova.apply(lambda row:
                                           pos_estim(row.condition),
                                           axis=1)
# Transformo los valores a tipo numérico (para que pueda recibirla la función
# de pingouin)
data_anova['value'] = pd.to_numeric(data_anova['value'])

# Hago el test de ANOVA de dos factores y medidas repetidas.
# Lo que nos interesa de tabla_anova es 'F' y 'p-unc'.
tabla_anova = pg.rm_anova(dv='value',
                          within=['par_sf', 'pos_estim'],
                          subject='index',
                          data=data_anova)

# Plot de interacción entre factores.
# Nota: existe from statsmodels.graphics.factorplots import interaction_plot
# que hace exactamente el gráfico que queremos, pero ponerle barras de error
# es un bardo (hay que modificar la librería). Pero me parece útil tenerlo a
# a mano para poder visualizar rápido la interacción entre factores.
plt.figure(figsize=(10, 8))
plt.errorbar([0, 1],
             [meandata['LL'].mean(), meandata['LR'].mean()],
             [meandata['LL'].std()/np.sqrt(len(meandata['LL'])),
              meandata['LR'].std()/np.sqrt(len(meandata['LR']))],
             color='k',
             fmt='s-',
             linewidth=3,
             label='Estimulo Izq',
             capsize=10,
             markersize=12)
plt.errorbar([0, 1],
             [meandata['RR'].mean(), meandata['RL'].mean()],
             [meandata['RR'].std()/np.sqrt(len(meandata['RR'])),
              meandata['RL'].std()/np.sqrt(len(meandata['RL']))],
             fmt='^-',
             linewidth=3,
             label='Estimulo Der',
             capsize=10,
             color=cbPaleta[1],
             markersize=13)

# Para que en el plot aparezcan los datos de los sujetos
plt.plot(np.zeros(len(meandata)), meandata['LL'],
         's', color='gray', markersize=9)
plt.plot(np.zeros(len(meandata)), meandata['RR'],
         '^', color=cbPaleta[6], markersize=9)
plt.plot(np.ones(len(meandata)), meandata['LR'],
         's', color='gray', markersize=9)
plt.plot(np.ones(len(meandata)), meandata['RL'],
         '^', color=cbPaleta[6], markersize=9)

plt.xlabel('Par Estim-Fdbk', fontsize=15)
plt.ylabel('Asincronia promedio[ms]', fontsize=15)
labels = ['Mismo', 'Diferente']
plt.xticks([0, 1], labels, fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend(fontsize=15)
# plt.savefig('Factores_nuevos_MEAN_Defensa.png')

# Hago exactamente lo mismo para la desviación estándar
data_anovastd = pd.melt(stddata.loc[:, stddata.columns != 'BB'].reset_index(),
                        id_vars='index',
                        var_name=['condition'])

# Agrego una columna con la información del factor "Par estim-fdbk"
data_anovastd['par_sf'] = data_anovastd.apply(lambda row:
                                              par_stimfdbk(row.condition),
                                              axis=1)
# Agrego una columan con la información del factor "Posicion del estímulo"
data_anovastd['pos_estim'] = data_anovastd.apply(lambda row:
                                                 pos_estim(row.condition),
                                                 axis=1)
# Transformo los valores a tipo numérico (para que pueda recibirla la función
# de pingouin)
data_anovastd['value'] = pd.to_numeric(data_anovastd['value'])

# Hago el test de ANOVA de dos factores y medidas repetidas.
# Lo que nos interesa de tabla_anova es 'F' y 'p-unc'.
tabla_anovastd = pg.rm_anova(dv='value',
                             within=['par_sf', 'pos_estim'],
                             subject='index',
                             data=data_anovastd)

# Plot de interacción entre factores.
plt.figure(figsize=(10, 8))
plt.errorbar([0, 1],
             [stddata['LL'].mean(), stddata['LR'].mean()],
             [stddata['LL'].std()/np.sqrt(len(stddata['LL'])),
              stddata['LR'].std()/np.sqrt(len(stddata['LR']))],
             color='k',
             fmt='s-',
             linewidth=3,
             label='Estimulo Izq',
             capsize=10,
             markersize=12)
plt.errorbar([0, 1],
             [stddata['RR'].mean(), stddata['RL'].mean()],
             [stddata['RR'].std()/np.sqrt(len(stddata['RR'])),
              stddata['RL'].std()/np.sqrt(len(stddata['RL']))],
             fmt='^-',
             linewidth=3,
             label='Estimulo Der',
             capsize=10,
             color=cbPaleta[1],
             markersize=13)

# Para que en el plot aparezcan los datos de los sujetos
plt.plot(np.zeros(len(stddata)), stddata['LL'],
         's', color='gray', markersize=9)
plt.plot(np.zeros(len(stddata)), stddata['RR'],
         '^', color=cbPaleta[6], markersize=9)
plt.plot(np.ones(len(stddata)), stddata['LR'],
         's', color='gray', markersize=9)
plt.plot(np.ones(len(stddata)), stddata['RL'],
         '^', color=cbPaleta[6], markersize=9)

plt.xlabel('Par Estim-Fdbk', fontsize=15)
plt.ylabel('Desviación estándar promedio[ms]', fontsize=15)
labels = ['Mismo', 'Diferente']
plt.xticks([0, 1], labels, fontsize=15)
plt.yticks(fontsize=15)
plt.grid(True)
plt.legend(fontsize=15)
# plt.savefig('Factores_nuevos_STD_Defensa.png')

# %% TEST DE STUDENT ENTRE (BB,RR) Y (BB,LL), PAREADO.

# Paired Student's t-test
# Defino alpha
alpha = 0.05/2

# Comparo BB con LL en los valores de asincronía promedio
statBBLL, pBBLL = ttest_rel(meandata['BB'], meandata['LL'])
print('Statistics=%.3f, p=%.3f' % (statBBLL, pBBLL))

if pBBLL > alpha:
    print('Accept null hypothesis that the means between BB and LL are equal.')
else:
    print('Reject null hypothesis that the means between BB and LL are equal.')

# Comparo BB con RR en los valores de asincronía promedio
statBBRR, pBBRR = ttest_rel(meandata['BB'], meandata['RR'])
print('Statistics=%.3f, p=%.3f' % (statBBRR, pBBRR))

if pBBRR > alpha:
    print('Accept null hypothesis that the means between BB and RR are equal.')
else:
    print('Reject null hypothesis that the means between BB and RR are equal.')


# Lo mismo pero con la desviación estándar
# Comparo BB con LL en desviación estándar promedio
stdBBLL, stdpBBLL = ttest_rel(stddata['BB'], stddata['LL'])
print('Statistics=%.3f, p=%.3f' % (stdBBLL, stdpBBLL))

if stdpBBLL > alpha:
    print('Accept null hypothesis that the means between BB and LL'
          + ' (STD) are equal.')
else:
    print('Reject null hypothesis that the means between BB and LL'
          + ' - STD are equal.')

# Comparo BB con LL en desviación estándar promedio
stdBBRR, stdpBBRR = ttest_rel(stddata['BB'], stddata['RR'])
print('Statistics=%.3f, p=%.3f' % (stdBBRR, stdpBBRR))

if stdpBBRR > alpha:
    print('Accept null hypothesis that the means between BB and RR'
          + '(STD) are equal.')
else:
    print('Reject null hypothesis that the means between BB and RR'
          + '(STD) are equal.')
