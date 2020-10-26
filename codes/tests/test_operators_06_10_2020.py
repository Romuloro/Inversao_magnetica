import pytest
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, aux_operators_array

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
                'n_dip': 10,
                'n_pop': 20,
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

populacao = Operators_array.create_population(**population)
print("População Inicial: {}".format(populacao))
print("\n")

val_fit = []
ind_better = []
anomaly_better = []

for t in range(15000):
    fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    anomaly_better.append(anomaly[min_fit])
    val_fit.append(min(fit_))
    pais_, escolhidos = Operators_array.tournament_selection(populacao, fit_)
    filho_ = Operators_array.crossover_elitista(pais_, escolhidos, fit_)
    filho_ = Operators_array.mutacao_vhomo(filho_, **filhos_mut)
    populacao = Operators_array.elitismo(populacao, filho_, fit_)


fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')


print(min(val_fit))
