import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, aux_operators_array, graphs_and_dist, genetic_algorithm

acquisition = {'nx': 20,
                  'ny': 20,
                  'xmin': -5000,
                  'xmax': 5000,
                  'ymin': -5000,
                  'ymax': 5000,
                  'z': -50.0,
                  'color': '.r'}


x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)


data_cubo = pd.read_table('Logfile/30_10_2020_11_59/data_mag.cvs', sep =',')
anomaly_cubo = np.reshape(np.array(data_cubo['Anomalia Magnética(nT)']), (20,20))



#plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 7000.0,
                'z_min': 0.0,
                'n_dip': 20,
                'n_pop': 20,
                'inclmax': 5.0,
                'inclmin': -5.0,
                'declmax': 5.0,
                'declmin': -5.0,
                'magmax': 2.5,
                'magmin': 1.5,
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
                'magmax': 2.5,
                'magmin': 1.5,
                'homogeneo': True
                }

ini = time.time()

populacao = Operators_array.create_population(**population)
#print("População Inicial: {}".format(populacao))
#print("\n")

val_fit = []
ind_better = []
anomaly_better = []
val_theta = []
final_pop = []

for t in range(1000):
    '''if t % 2:
        theta, MST, anomaly = graphs_and_dist.theta_value(populacao, X, Y, Z, I, D)
        min_theta = theta.index(min(theta))
        ind_better.append(populacao[min_theta])
        anomaly_better.append(anomaly[min_theta])
        val_theta.append(min(theta))
        pais_, escolhidos = Operators_array.tournament_selection(populacao, theta)
        filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
        filho_ = Operators_array.mutacao_vhomo(filho_, **filhos_mut)
        populacao = Operators_array.elitismo(populacao, filho_, theta)
        print('geracao', t)
        print(val_theta[int(t/2)])
        #for i in range(3):
            #print(val_theta[i])
        #for i in val_theta:
            #print(i)'''
    
    #else:
    fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    anomaly_better.append(anomaly[min_fit])
    val_fit.append(min(fit_))
    pais_, escolhidos = Operators_array.tournament_selection(populacao, fit_)
    filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
    filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut)
    populacao = Operators_array.elitismo(populacao, filho_, fit_)
    print('geracao', t)
    print(val_fit[t])#[int(t/2)])
        #for i in range(3):
            #print(val_fit[i])
        #for i in val_fit:
            #print(i)




fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')
final_pop.append(populacao)

print(min(val_fit))
#print(min(val_theta))

