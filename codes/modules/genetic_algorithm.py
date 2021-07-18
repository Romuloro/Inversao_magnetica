import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from numba import jit
from numba.typed import List

a = sys.path.append('../modules/')
a = sys.path.append('../codes/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, aux_operators_array, graphs_and_dist

acquisition = {'nx': 20,
               'ny': 20,
               'xmin': -6000,
               'xmax': 6000,
               'ymin': -6000,
               'ymax': 6000,
               'z': -50.0,
               'color': '.r'}

x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)

os.chdir('/home/romulo/my_project_dir/Inversao_magnetica/codes/tests/Logfile')
data_cubo = pd.read_table('15_07_2021_18_10/data_mag.cvs', sep=',')
anomaly_cubo = np.reshape(np.array(data_cubo['Anomalia Magnética(nT)']), (20, 20))

momento = 1.035e10 / 20  # 3.8X10^10/ndip
# print(momento)

# plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 6000.0,
              'xmin': -6000.0,
              'ymax': 6000.0,
              'ymin': -6000.0,
              'zlim': 7000.0,
              'z_min': 0.0,
              'n_dip': 20,
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

filhos_mut = {'xmax': 6000.0,
              'xmin': -6000.0,
              'ymax': 6000.0,
              'ymin': -6000.0,
              'zlim': 7000.0,
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
# print("População Inicial: {}".format(populacao))
# print("\n")


n = 3000
lamb = 1.25e-2


def ga(lamb, n, anomaly_cubo, filhos_mut, population):

    populacao = Operators_array.create_population(**population)

    val_fit = List()
    val_phi = List()
    ind_better = List()
    anomaly_better = List()
    # final_pop = List()
    val_theta = List()
    # ind_theta = List()
    incl_better = List()
    decl_better = List()
    diversity_x = List()
    diversity_y = List()
    diversity_z = List()
    diversity_incl = List()
    diversity_decl = List()
    diversity_mom = List()

    for t in range(n):
        populacao = List(populacao)
        mean_x, std_x = aux_operators_array.deversity_calculate(populacao, 'X')
        mean_y, std_y = aux_operators_array.deversity_calculate(populacao, 'Y')
        mean_z, std_z = aux_operators_array.deversity_calculate(populacao, 'Z')
        mean_incl, std_incl = aux_operators_array.deversity_calculate(populacao, 'incl')
        mean_decl, std_decl = aux_operators_array.deversity_calculate(populacao, 'decl')
        mean_mom, std_mom = aux_operators_array.deversity_calculate(populacao, 'mom')
        diversity_x.append(std_x)
        diversity_y.append(std_y)
        diversity_z.append(std_z)
        diversity_incl.append(std_incl)
        diversity_decl.append(std_decl)
        diversity_mom.append(std_mom)
        normal_gama, gama, anomaly, MST, theta, phi = Operators_array.final_fit(X, Y, Z, I, D, populacao, anomaly_cubo, lamb=lamb)
        # fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
        # theta, MST = graphs_and_dist.theta_value(populacao)
        min_fit = normal_gama.index(min(normal_gama))
        ind_better.append(populacao[min_fit])
        anomaly_better.append(anomaly[min_fit])
        val_fit.append(min(gama))
        val_theta.append(min(theta))
        val_phi.append(min(phi))
        incl_better.append(populacao[min_fit][len(ind_better[0]) - 1, 0])
        decl_better.append(populacao[min_fit][len(ind_better[0]) - 1, 1])
        pais_ = Operators_array.tournament_selection_ranking_diversit(populacao, normal_gama)
        filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
        if (t >= 5) and (val_fit[t] == val_fit[t - 5]):
            filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut, prob_mut=0.4)  # aumenta mut para X
        else:
            filho_ = Operators_array.mutacao_multi_vhomo(filho_, **filhos_mut)  # manter mut em 0.05
        populacao = Operators_array.elitismo(populacao, filho_, normal_gama, n_fica=5)
        # populacao = Operators_array.elitismo_c_violation(populacao, filho_, theta, n_fica = 5)

    return populacao, anomaly_better, ind_better, val_fit, val_phi, val_theta, incl_better, decl_better, diversity_x, diversity_y, diversity_z, diversity_incl, diversity_decl, diversity_mom

