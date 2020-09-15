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
                  'xmin': -10000,
                  'xmax': 10000,
                  'ymin': -10000,
                  'ymax': 10000,
                  'z': -100.0,
                  'color': '.r'}

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

incl, decl, mag = sample_random.sample_random_mag(**mag_bounds)

x, y, X, Y, Z = plot_3D.create_aquisicao(**acquisition)

balls_mag = {'incl': incl,
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

tfa_n_bolinhas = sample_random.tfa_n_dots(**balls_mag)

plot_3D.modelo_anomalia_3D(Y, X, tfa_n_bolinhas, coodY, coodX, coodZ, mag)

population = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 150.0,
                'z_min': 0.0,
                'n_dip': 6,
                'n_pop': 4,
                'inclmax': -80.0,
                'inclmin': 15.0,
                'declmax': 172.0,
                'declmin': -15.0,
                'magmax': 5.0,
                'magmin': 0.5,
                'homogeneo': True
                }

I, D = 30.0, 50.0

filhos_mut = {'xmax': 5000.0,
                'xmin': -5000.0,
                'ymax': 5000.0,
                'ymin': -5000.0,
                'zlim': 150.0,
                'z_min': 0.0,
                'n': 1,
                'inclmax': -80.0,
                'inclmin': 15.0,
                'declmax': 172.0,
                'declmin': -15.0,
                'magmax': 5.0,
                'magmin': 0.5,
                'homogeneo': True
                }

ini = time.time()

populacao = Operators.create_population(**population)
print("População Inicial: {}".format(populacao))
print("\n")

val_fit = []

for t in range(1000):
    fit_ = Operators.fit_value(X, Y, Z, I, D, populacao, tfa_n_bolinhas)
    pais_ = Operators.tournament_selection(populacao, fit_)
    filho_ = Operators.crossover(pais_)
    filho_ = Operators.mutacao(filho_, **filhos_mut)
    populacao = Operators.elitismo(populacao, filho_, fit_)

    fit_cont = Operators.fit_value(X, Y, Z, I, D, populacao, tfa_n_bolinhas)
    min_fit = min(fit_cont)
    val_fit.append(min_fit)

fim = time.time()
print(f'Tempo do algoritmo genético: {fim-ini}')

last_fit = Operators.fit_value(X, Y, Z, I, D, populacao, tfa_n_bolinhas)
print(min(last_fit))

print(val_fit)