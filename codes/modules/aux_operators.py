import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
a = sys.path.append('../modules/')
import Operators as top
import sample_random


def definition_prob(pai_torneio, X, Y, Z, I, D, n_filhos, tfa_n_bolinhas):
    prob_pai = []
    prob_mae = []
    sum_den = []

    fit_pais = top.fit_value(X, Y, Z, I, D, pai_torneio, tfa_n_bolinhas)
    f_pais, f_maes = fit_pais[0:n_filhos], fit_pais[n_filhos:len(pai_torneio)]
    for i in range(n_filhos):
        casal = [f_pais[i], f_maes[i]]
        melhor = casal.index(min(casal))
        if melhor == 0:
            prob_pai.append(random.uniform(0.5, 1.0))
            prob_mae.append(random.random())
        else:
            prob_mae.append(random.uniform(0.5, 1.0))
            prob_pai.append(random.random())
        sum_den.append(prob_pai[i] + prob_mae[i])

    return prob_pai, prob_mae, sum_den


def caculation_anomaly(X, Y, Z, I, D, pop):
    raio = 100.0
    anomaly = []

    for i in range(len(pop)):
        spheres = []
        incl = []
        decl = []
        mag = []
        for j in range(len(pop[0])):
            spheres.append((pop[i][j][0], pop[i][j][1], pop[i][j][2], raio))
            incl.append(pop[i][j][3])
            decl.append(pop[i][j][4])
            mag.append(pop[i][j][5])
        anomaly.append(sample_random.tfa_n_dips(incl, decl, mag, len(pop[0]), X, Y, Z, I, D, spheres))
    return anomaly


def caculation_onlyone_anomaly(X, Y, Z, I, D, pop):

    raio = 100.0
    spheres = []
    incl = []
    decl = []
    mag = []
    for j in range(len(pop)):
        spheres.append((pop[j][0], pop[j][1], pop[j][2], raio))
        incl.append(pop[j][3])
        decl.append(pop[j][4])
        mag.append(pop[j][5])
    anomaly = (sample_random.tfa_n_dips(incl, decl, mag, len(pop), X, Y, Z, I, D, spheres))

    return anomaly

