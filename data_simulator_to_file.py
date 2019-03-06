import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import OrderedDict
import time
import datetime

# rise function
def feature_up(min, max, N, cte_t, noise_amplitude):
    x = np.arange(1.0, N+1)
    y = min + (max - min)*np.exp(-(N - x)/cte_t)
    y = y + noise_amplitude*np.random.normal(size=N)
    return y

# down function
def feature_down(max, min, N, cte_t, noise_amplitude):
    x = np.arange(1.0, N+1)
    y = max - (max - min)*np.exp(-(N - x)/cte_t)
    y = y + noise_amplitude*np.random.normal(size=N)
    return y

# dual function
def feature_dual(min_ciclo1, max_ciclo1, threshold, N, cte_t, noise_amplitude):
    # 0 < threshold < 1
    x = np.arange(1.0, N+1)
    val_ciclo1 = np.array([np.random.uniform(min_ciclo1, max_ciclo1)])
    limite = min_ciclo1 + (max_ciclo1 - min_ciclo1)*threshold
    rango_cicloN = 0.025*np.mean([min_ciclo1, max_ciclo1])
    if val_ciclo1 <= limite:
        # down
        y = val_ciclo1 - rango_cicloN*threshold*np.random.uniform(0, 1)*np.exp(-(N - x)/cte_t)
    else:
        # up
        y = val_ciclo1 + rango_cicloN*(1 - threshold)*np.random.uniform(0, 1)*np.exp(-(N - x)/cte_t)
    
    y = y + noise_amplitude*np.random.normal(size=N)
    return y


# 7 FEATURES: 4 rising (una discreta), 2 decreasing, 1 dual
MIN_MAX = np.array([[641.0, 644.0], 
                   [555.0, 551.0],
                   [46.8 ,  48.5],
                   [388.0, 400.0],
                   [1380.0, 1440.0],
                   [39.5, 38.1],
                   [8110.0, 8150.0]])

# INITIALIZE THE DATASET
dataset = pd.DataFrame()

N_features = MIN_MAX.shape[0]
window_TTF = 10 # time-to-failure (in number of cycles)
N_ENGINES = 6

for engine_ID in range(1, N_ENGINES + 1):
    
    #N = np.random.randint(100, 300) # failures (dist. uniform), different for each ENGINE
    N = round(np.random.triangular(120, 180, 300)) # failures (dist. triang.), different for each ENGINE
    matrix_features = np.zeros((N, N_features + 4))
    matrix_features[:, 0] = engine_ID
    matrix_features[:, 1] = np.arange(1, N+1)
    matrix_features[:, N_features + 2] = N - matrix_features[:, 1] # RUL
    iterable = (1 if rul < window_TTF else 0 for rul in matrix_features[:, N_features + 2])
    matrix_features[:, N_features + 3] = np.fromiter(iterable, 'int32') # TTF

    # FEATURE 1: RISING
    i = 0
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = feature_up(np.amin(min_max), np.amax(min_max), N, N*0.2, noise_amplitude)


    # FEATURE 2: DECREASING
    i = 1
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = feature_down(np.amax(min_max), np.amin(min_max), N, N*0.2, noise_amplitude)

    # FEATURE 3: RISING
    i = 2
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = feature_up(np.amin(min_max), np.amax(min_max), N, N*0.2, noise_amplitude)

    # FEATURE 4: RISING
    i = 3
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = np.rint(feature_up(np.amin(min_max), np.amax(min_max), N, N*0.2, noise_amplitude))

    # FEATURE 5: RISING
    i = 4
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = feature_up(np.amin(min_max), np.amax(min_max), N, N*0.2, noise_amplitude)

    # FEATURE 6: DECREASING
    i = 5
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    peak_min = min_ciclo1 + 0*(max_ciclo1 - min_ciclo1)
    min_i = np.random.triangular(min_ciclo1, peak_min, 0.5*(min_ciclo1 + max_ciclo1))
    peak_max = min_ciclo1 + 1*(max_ciclo1 - min_ciclo1)
    max_i = np.random.triangular(0.5*(min_ciclo1 + max_ciclo1), peak_max, max_ciclo1)
    min_max = np.array([min_i, max_i])
    noise_amplitude = 0.15*np.absolute(np.amax(min_max) - np.amin(min_max))
    matrix_features[:, i+2] = feature_down(np.amax(min_max), np.amin(min_max), N, N*0.2, noise_amplitude)

    # FEATURE 7: DUAL = RISING OR DECREASING
    i = 6
    min_ciclo1 = np.amin(MIN_MAX[i, :])
    max_ciclo1 = np.amax(MIN_MAX[i, :])
    noise_amplitude = 0.05*np.absolute(max_ciclo1 - min_ciclo1)
    matrix_features[:, i+2] = feature_dual(min_ciclo1, max_ciclo1, 0.2, N, N*0.2, noise_amplitude)

    
    # DATAFRAME
    df = pd.DataFrame(matrix_features)
    df[[0, 1, N_features + 2, N_features + 3]] = df[[0, 1, N_features + 2, N_features + 3]].astype('int32')

    """
    # FIGURES: CURRENT ENGINE
    fig, ax = plt.subplots(figsize=(20, 10))
    df.loc[:, 2:(N_features + 1)].plot(kind='line', subplots=True, layout=(3, 3), sharex=True, ax=ax)
    plt.show()
    """

    dataset = dataset.append(df, ignore_index=True)

# REPLACE TTF LABELS
dataset.loc[:, N_features + 3].replace({0: 'long', 1: 'short'}, inplace=True)
header = ['engine_ID', 'cycle', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 'RUL', 'TTF']
dataset.columns = header
dataset.head()

# SAVE TO FILE
date_time_now = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
filename = 'D:/data_sim_' + date_time_now + '.csv'
dataset.to_csv(filename, index=False, header=header)

# FIGURES: ALL ENGINES
"""
for i in range(2, 9):
    fig, ax = plt.subplots(figsize=(20, 10))
    dataset.plot(kind='scatter', x=1, y=i, ax=ax, color='r')
    plt.show()
"""

# STREAMING SIMULATION ON SCREEN
dataset_dict = dataset.to_dict(orient='records', into=OrderedDict)
freq = 10 # samples per second

for i in range(dataset.shape[0]):
    datax = json.dumps(dataset_dict[i])
    print(datax)
    time.sleep(1/freq)
