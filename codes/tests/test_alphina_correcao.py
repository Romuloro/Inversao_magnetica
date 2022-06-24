import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import math
import time
a = sys.path.append('../modules/') # endereco das funcoes implementadas por voce!
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, Operators_alpina
#import test_operators_06_10_2020 as top
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection




population = {'xmax': 10.0,
                'xmin': 0.0,
                'ymax': 10.0,
                'ymin': 0.0,
                'zlim': 10.0,
                'z_min': 0.0,
                'n_dip': 1,
                'n_pop': 10,
                'homogeneo': True
                }

filhos_mut = {'xmax': 10.0,
                'xmin': 0.0,
                'ymax': 10.0,
                'ymin': 0.0,
                'zlim': 10.0,
                'z_min': 0.0,
                'n': 1,
                'homogeneo': True
                }

pop_inicial = Operators_alpina.create_population(**population)


x = np.linspace(0, 10, 20, endpoint=True)
y = np.linspace(0, 10, 20, endpoint=True)
#----------------------------------------------------------------------------------------------------#
X,Y = np.meshgrid(x,y)
Z = np.copy(X)*0.0 + 10
#----------------------------------------------------------------------------------------------------#
plt.figure(figsize=(12,12))
plt.title('Levantamento aéreo')
plt.plot(X,Y, '.r')
plt.show()

function = Operators_alpina.function_alphina_normal(X, Y)

populacao = [ ]
valor_fit = []
anomaly_better = []
final_pop = []
populacao.append( pop_inicial )
p = list( np.copy( pop_inicial ) )
for t in range( 3 ):
    temp, vfit = Operators_alpina.ag( p,**filhos_mut  )
    print('******************')
    valor_fit.append(vfit)
    populacao.append( temp )
    print(populacao[t+1])
    print('******************')
    p = temp

print('Pop é :')
print(populacao[1])
print('***********************************')
print(populacao[2])
print('***********************************')
print(populacao[3])
print('***********************************')
#print(populacao[5])
#print('***********************************')
