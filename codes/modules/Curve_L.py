import warnings

import numpy as np

warnings.filterwarnings("ignore")
import sys
a = sys.path.append('../modules/')
c = sys.path.append('../tests/')
import os
import pandas as pd
import genetic_algorithm as top


def curve_L():   
    lamb = top.lamb
    n = top.n
    n_ga = 10
    anomaly_cubo = top.anomaly_cubo
    filhos_mut = top.filhos_mut
    population = top.population
    result = np.zeros((n_ga, 3))

    for i in range(n_ga):
        lamb = lamb * (1e1**(i))      
        populacao, anomaly_better, ind_better, val_fit, val_phi, val_theta, incl_better, decl_better, mom_better, diversity_x, diversity_y, diversity_z, diversity_incl, diversity_decl, diversity_mom, z_better = top.ga(lamb, n, anomaly_cubo, filhos_mut, population)
        
        result[i,0], result[i,1], result[i,2] = val_phi[len(val_phi)-1], val_theta[len(val_theta)-1], lamb
        print(result)
        lamb = top.lamb

    print(result)
    return result


#result = curve_L()

#os.chdir('/home/romulo/my_project_dir/Inversao_magnetica/codes/tests/Curva_L')
#endereco = './test_1'
#os.mkdir(endereco)

#curve_L_result = pd.DataFrame(data = result)
#header = ['phi, theta, lambda']
#curve_L_result.to_csv('/home/romulo/my_project_dir/Inversao_magnetica/codes/tests//Curva_L/test_1/result_1.cvs', index = False, header = header)


