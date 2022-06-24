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
                'n_dip': 2,
                'n_pop': 10,
                'homogeneo': True
                }

filhos_mut = {'xmax': 10.0,
                'xmin': 0.0,
                'ymax': 10.0,
                'ymin': 0.0,
                'zlim': 100.0,
                'z_min': 0.0,
                'n': 1,
                'homogeneo': True
                }

pop_inicial = Operators_alpina.create_population(**population)

mae1 = np.random.rand(2,2)
print(mae1)
print(type(mae1))
print('-------------')
pai1 = np.random.rand(2,2)
print(pai1)


p_shape = tuple(pop_inicial[0].shape) #tuple(mae1.shape)
probs = np.random.rand(p_shape[0], p_shape[1])
filhos = []
n_filhos = int(len(pop_inicial) / 2)
pai = pop_inicial[0:n_filhos]
mae = pop_inicial[n_filhos:len(pop_inicial)]
for k in range(n_filhos):
    i_pai = pai[k]
    i_mae = mae[k]
    m_mae = np.zeros(p_shape)
    m_pai = np.zeros(p_shape)
    for i in range(p_shape[0]):
        for j in range(p_shape[1]):
            if probs[i,j] < 0.5:
                m_pai[i,j], m_mae[i,j] = i_mae[i,j], i_pai[i,j]
            else:
                m_pai[i, j], m_mae[i, j] = i_pai[i, j], i_mae[i, j]
    filhos +=[m_pai, m_mae]



