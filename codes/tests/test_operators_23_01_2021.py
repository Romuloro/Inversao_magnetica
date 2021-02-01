import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, aux_operators_array, graphs_and_dist

acquisition = {'nx': 20,
                  'ny': 20,
                  'xmin': -5000,
                  'xmax': 5000,
                  'ymin': -5000,
                  'ymax': 5000,
                  'z': -50.0,
                  'color': '.r'}


x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)


data_cubo = pd.read_table('Logfile/28_01_2021_16_20/data_mag.cvs', sep =',')
anomaly_cubo = np.reshape(np.array(data_cubo['Anomalia Magnética(nT)']), (20,20))

#momento = 38000000000

#plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 7000.0,
                'z_min': 0.0,
                'n_dip': 25,
                'n_pop': 20,
                'inclmax': 5.0,
                'inclmin': -5.0,
                'declmax': 5.0,
                'declmin': -5.0,
                'mmax': 100000000000,
                'mmin': 10000000,
                'homogeneo': True
                }

I, D = 5.0, 70.0

filhos_mut = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 7000.0,
                'z_min': 0.0,
                'n': 1,
                'inclmax': 5.0,
                'inclmin': -5.0,
                'declmax': 5.0,
                'declmin': -5.0,
                'magmax': 100000000000,
                'magmin': 1000000000,
                'homogeneo': True
                }

ini = time.time()

populacao = Operators_array.create_population(**population)
#print("População Inicial: {}".format(populacao))
#print("\n")

val_fit = []
ind_better = []
anomaly_better = []
final_pop = []
val_theta = []
ind_theta = []

for t in range(2000):
    fit_, anomaly, MST, theta = Operators_array.final_fit(X, Y, Z, I, D, populacao, anomaly_cubo, lamb = 0.00005)
    #fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    anomaly_better.append(anomaly[min_fit])
    val_fit.append(min(fit_))
    min_theta = theta.index(min(theta))
    ind_theta.append(populacao[min_theta])
    val_theta.append(min(theta))
    pais_, escolhidos = Operators_array.tournament_selection(populacao, fit_)
    filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
    if (t >= 5) and (val_fit[t] == val_fit[t-5]):
        filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut, prob_mut = 0.4) #aumenta mut para X
    else:
        filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut) #manter mut em 0.05
    populacao = Operators_array.elitismo(populacao, filho_, fit_, n_fica = 5)
    print('geracao', t)
    print(val_fit[t])




fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')
final_pop.append(populacao)

print(min(val_fit))
