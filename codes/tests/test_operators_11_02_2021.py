import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
from numba import jit
from numba.typed import List
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

momento = 38000000000/30 #3.8X10^10/ndip
#print(momento)

#plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 5000.0,
                'z_min': 0.0,
                'n_dip': 30,
                'n_pop': 50,
                'inclmax': 5.0,
                'inclmin': -5.0,
                'declmax': 5.0,
                'declmin': -5.0,
                'mmax': momento,
                'mmin': momento,
                'homogeneo': True
                }

I, D = 5.0, 70.0

filhos_mut = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 5000.0,
                'z_min': 0.0,
                'n': 1,
                'inclmax': 5.0,
                'inclmin': -5.0,
                'declmax': 5.0,
                'declmin': -5.0,
                'magmax': momento,
                'magmin': momento,
                'homogeneo': True
                }

ini = time.time()

populacao = Operators_array.create_population(**population)
#print("População Inicial: {}".format(populacao))
#print("\n")

val_fit = List()
val_phi = List()
ind_better = List()
anomaly_better = List()
final_pop = List()
val_theta = List()
ind_theta = List()
incl_better = List()
decl_better = List()

n = 5000

for t in range(n):
    populacao = List(populacao)
    gama, anomaly, MST, theta, phi = Operators_array.final_fit(X, Y, Z, I, D, populacao, anomaly_cubo, lamb = 0.00005)
    #fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
    #theta, MST = graphs_and_dist.theta_value(populacao)
    min_fit = gama.index(min(gama))
    ind_better.append(populacao[min_fit])
    anomaly_better.append(anomaly[min_fit])
    val_fit.append(min(gama))
    val_theta.append(min(theta))
    val_phi.append(min(phi))
    incl_better.append(populacao[min_fit][len(ind_better[0]) - 1, 0])
    decl_better.append(populacao[min_fit][len(ind_better[0]) - 1, 1])
    pais_ = Operators_array.tournament_selection_ranking_diversit(populacao, gama)
    filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
    if (t >= 5) and (val_fit[t] == val_fit[t-5]):
        filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut, prob_mut = 0.4) #aumenta mut para X
    else:
        filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut) #manter mut em 0.05
    populacao = Operators_array.elitismo(populacao, filho_, gama, n_fica = 5)
    #populacao = Operators_array.elitismo_c_violation(populacao, filho_, theta, n_fica = 5)
    print('geracao', t)
    print(val_fit[t])




fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')
final_pop.append(populacao)

print('O menor fit da última geração é:',min(val_fit))
m_err = aux_operators_array.relative_error(ind_better[n-1][len(ind_better[n-1])-1,2], momento)
incl_err = aux_operators_array.relative_error(ind_better[n-1][len(ind_better[n-1])-1,0], 0.0)
decl_err = aux_operators_array.relative_error(ind_better[n-1][len(ind_better[n-1])-1,1], 0.0)
print('O erro relativo do momento de dipolo é:', m_err)
print('O erro relativo do inclinação magnética é:', incl_err)
print('O erro relativo do declinação magnética é:', decl_err)

