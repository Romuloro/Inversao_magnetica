import pytest
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators, aux_operators

acquisition = {'nx': 100,
                  'ny': 100,
                  'xmin': -5000,
                  'xmax': 5000,
                  'ymin': -5000,
                  'ymax': 5000,
                  'z': -100.0,
                  'color': '.r'}
"""
cood_bounds = {'xmax': 250.0,
                'xmin': 40.0,
                'ymax': 100.0,
                'ymin': 0.0,
                'zlim': 150,
                'z_min': 100,
                'n': 5}

mag_bounds = {'inclmax': -20.0,
                'inclmin': 15.0,
                'declmax': 20.0,
                'declmin': -15.0,
                'magmax': 5.0,
                'magmin': 1.0,
                'homogeneo': True,
                'n': 5}

coodX, coodY, coodZ = sample_random.sample_random_coordinated(**cood_bounds)

incl, decl, mag = sample_random.sample_random_mag(**mag_bounds)"""

x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)

"""balls_mag = {'incl': incl,
               'decl': decl,
               'mag': mag,
               'n': 5,
               'Xref': X,
               'Yref': Y,
               'Zref': Z,
               'I': 30.0,
               'D': 50.0,
               'coodX': coodX,
               'coodY': coodY,
               'coodZ': coodZ,
               'raio': 100.0}

tfa_n_bolinhas = sample_random.tfa_n_dots(**balls_mag)"""

data_cubo = pd.read_table('Logfile/20_09_2020_11_26/data_mag.cvs', sep =',')
anomaly_cubo = np.reshape(np.array(data_cubo['Anomalia Magnética(nT)']), (100,100))


plt.figure(figsize=(9,10))
plt.contourf(Y, X, anomaly_cubo, 20, cmap = plt.cm.RdBu_r)
plt.title('Anomalia de Campo Total(nT)', fontsize = 20)
plt.xlabel('East (m)', fontsize = 20)
plt.ylabel('North (m)', fontsize = 20)
plt.colorbar()
#plt.savefig('prisma_anomalia.pdf', format='pdf')
plt.show()


#plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 1500.0,
                'z_min': 500.0,
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
                'zlim': 1500.0,
                'z_min': 500.0,
                'n': 1,
                'inclmax': -40.0,
                'inclmin': -20.0,
                'declmax': -20.0,
                'declmin': 0.0,
                'magmax': 5.0,
                'magmin': 0.5,
                'homogeneo': True
                }

ini = time.time()

populacao = Operators.create_population(**population)
print("População Inicial: {}".format(populacao))
print("\n")

val_fit = []
ind_better = []
anomaly_better = []

for t in range(15000):
    fit_ = Operators.fit_value_v2(X, Y, Z, I, D, populacao, anomaly_cubo)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    anomaly = aux_operators.caculation_onlyone_anomaly(X, Y, Z, I, D, populacao[min_fit])
    anomaly_better.append(anomaly)
    pais_ = Operators.tournament_selection(populacao, fit_)
    filho_ = Operators.crossover_eletista(pais_, X, Y, Z, I, D, anomaly_cubo)
    filho_ = Operators.mutacao_vhomo(filho_, **filhos_mut)
    populacao = Operators.elitismo(populacao, filho_, fit_)

    val_fit.append(min(fit_))

fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')

last_fit = Operators.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
print(min(last_fit))
