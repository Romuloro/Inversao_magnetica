import pytest
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators

acquisition = {'nx': 100,
                  'ny': 100,
                  'xmin': -5000,
                  'xmax': 5000,
                  'ymin': -5000,
                  'ymax': 5000,
                  'z': -100.0,
                  'color': '.r'}

x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)

data_cubo = pd.read_table('Logfile/20_09_2020_11_26/data_mag.cvs', sep =',')
anomaly_cubo = np.reshape(np.array(data_cubo['Anomalia Magnética(nT)']), (100,100))

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 1100.0,
                'z_min': 900.0,
                'n_dip': 6,
                'n_pop': 30,
                'inclmax': -40.0,
                'inclmin': -20.0,
                'declmax': -20.0,
                'declmin': 0.0,
                'magmax': 5.0,
                'magmin': 0.5,
                'homogeneo': True
                }

I, D = -30.0, -23.0

filhos_mut = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 1100.0,
                'z_min': 900.0,
                'n': 1,
                'inclmax': -40.0,
                'inclmin': -20.0,
                'declmax': -20.0,
                'declmin': 0.0,
                'magmax': 5.0,
                'magmin': 0.5,
                'homogeneo': True
                }

populacao = Operators.create_population(**population)
ind_better = []
fit_ = Operators.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
min_fit = fit_.index(min(fit_))
ind_better.append(populacao[min_fit])
pais_ = Operators.tournament_selection(populacao, fit_)

pega_filho = []
for t in range(10):
    filho_ = Operators.crossover(pais_)
    filho_ = Operators.mutacao_vhomo(filho_, **filhos_mut)
    pega_filho.append(filho_)

print(pega_filho)

print('Final')
