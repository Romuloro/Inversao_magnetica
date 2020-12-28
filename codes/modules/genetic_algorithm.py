import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
import time
a = sys.path.append('../modules/')
import plot_3D, auxiliars, salve_doc, sphere, sample_random, Operators_array, aux_operators_array, graphs_and_dist


def fit_algorithm(X, Y, Z, I, D, populacao, anomaly_cubo, filhos_mut):
    val_fit = []
    ind_better = []
    anomaly_better = []
    populacao_ = 0.0
    fit_, anomaly = Operators_array.fit_value(X, Y, Z, I, D, populacao, anomaly_cubo)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    anomaly_better.append(anomaly[min_fit])
    val_fit.append(min(fit_))
    pais_, escolhidos = Operators_array.tournament_selection(populacao, fit_)
    filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
    filho_ = Operators_array.mutacao_vhomo(filho_, **filhos_mut)
    populacao_ = Operators_array.elitismo(populacao, filho_, fit_)
    populacao = populacao_

    return min_fit, ind_better, anomaly_better, val_fit, populacao


def graph_algorithm(X, Y, Z, I, D, populacao, anomaly_cubo, filhos_mut):
    val_theta = []
    ind_better = []
    anomaly_better = []
    populacao_ = 0.0
    theta, MST, anomaly = graphs_and_dist.theta_value(populacao, X, Y, Z, I, D)
    min_theta = theta.index(min(theta))
    ind_better.append(populacao[min_theta])
    anomaly_better.append(anomaly[min_theta])
    val_theta.append(min(theta))
    pais_, escolhidos = Operators_array.tournament_selection(populacao, theta)
    filho_ = Operators_array.crossover_polyamory(pais_)  # Operators_array.uniform_crossover(pais_)
    filho_ = Operators_array.mutacao_vhomo(filho_, **filhos_mut)
    populacao_ = Operators_array.elitismo(populacao, filho_, theta)
    populacao = populacao_

    return min_theta, ind_better, anomaly_better, val_theta, populacao


