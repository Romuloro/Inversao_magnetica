# --------------------------------------------------------------------------------------------------
# Title: Auxiliary Mag Codes
# Author: Rômulo Rodrigues de Oliveira
# Description: Source codes
# Collaboratores: Rodrigo Bijani
# -----------------------------------------------------------------------------------------



import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
a = sys.path.append('../modules/')
import Operators as top
import sphere


def tfa_n_dips(incl, decl, mag, n, Xref, Yref, Zref, I, D, spheres):
    """
    Função com o objetivo calcular a anomalia magnética de n bolinhas.

    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: tfa_n_dots(**dicionario).

    :param dicionario: incl - Lista com os valores de inclinação magnética.
                       decl - Lista com os valores de declinação magnética.
                       mag - Lista com os valores de magnetização.
                       n - número de bolinhas desejadas.
                       Xref - Matrix com as coordenadas em X.
                       Yref - Matrix com as coordenadas em Y.
                       Zref - Matrix com as coordenadas em Z.
                       I - valor de inclinação regional.
                       D - valor de declinação regional.
                       spheres - Lista com os valores de coordenadas e raio de cada dipolo.
                           spheres[0] - Coordenada no eixo X.
                           spheres[1] - Coordenada no eixo Y.
                           spheres[2] - Coordenada no eixo Z.
                           spheres[3] - Raio do dipolo.
    :return: Uma matrix com os valores de anomália magnética para cada ponto do local estudado.
    """

    # ---------------------------------------------------------------------------------------------------------------------#
    tfa_n = 0
    raio = 500.0
    for i in range(n):
        tfa_cada = sphere.sphere_tfa(Xref, Yref, Zref, spheres[i], raio, mag, I, D, incl, decl)
        tfa_n += tfa_cada
    return tfa_n


def caculation_anomaly(X, Y, Z, I, D, pop):
    anomaly = []
    # n_dip = len(pop[0])-1

    for i in range(len(pop)):
        incl = pop[i][len(pop[0]) - 1, 0]
        decl = pop[i][len(pop[0]) - 1, 1]
        mag = pop[i][len(pop[0]) - 1, 2]
        spheres = pop[i][0:len(pop[0]) - 1, :]
        anomaly.append(tfa_n_dips(incl, decl, mag, len(pop[0]) - 1, X, Y, Z, I, D, spheres))
    return anomaly


def caculation_onlyone_anomaly(X, Y, Z, I, D, pop):

    raio = 100.0
    spheres = []
    incl = pop[len(pop[0]) - 1, 0]
    decl = pop[len(pop[0]) - 1, 1]
    mag = pop[len(pop[0]) - 1, 2]
    for j in range(len(pop[0])-1):
        spheres.append((pop[j][0], pop[j][1], pop[j][2], raio))
    anomaly = (tfa_n_dips(incl, decl, mag, len(pop[0])-1, X, Y, Z, I, D, spheres))

    return anomaly



def count_index_fit(lista):
    escolhido = []
    for i in range(len(lista)):
        n = lista[i]
        B = len([i for i in lista[0:i] if i <= n])
        n += B
        escolhido.append(n)
    return escolhido


def definition_prob(pai_torneio, escolhidos, fit, n_filhos):
    prob_pai = []
    prob_mae = []
    sum_den = []
    fit = np.array(fit)
    escolhidos = count_index_fit(escolhidos)

    fit_pais = list(fit[escolhidos])
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


def f_difference(dado_referencia, dado_calculado):
    """
    Função com o objetivo de calcular o valor da função diferença entre os dados de referência para os dados calculados.

    :param dado_referencia: O dado de referência.
    :param dado_calculado: O dado calculado que será comparado.
    :return: Valor da função diferença.
    """

    std = np.std(dado_referencia)
    dif = (dado_referencia - dado_calculado) ** 2 / (std ** 2)
    rms = np.sum(dif) / len(dif)

    return rms


