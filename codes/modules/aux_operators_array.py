# --------------------------------------------------------------------------------------------------
# Title: Auxiliary Mag Codes
# Author: Rômulo Rodrigues de Oliveira
# Description: Source codes
# Collaboratores: Rodrigo Bijani
# -----------------------------------------------------------------------------------------


from numba import jit
from numba.typed import List
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import pandas as pd
a = sys.path.append('../modules/')
import Operators as top
import sphere_teste


@jit(nopython=True)
def tfa_n_dips(incl, decl, m, n, Xref, Yref, Zref, I, D, spheres):
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
    tfa_n = np.zeros((len(Xref),len(Xref)))
    for i in range(n):
        tfa_cada = sphere_teste.sphere_tfa(Xref, Yref, Zref, spheres[i], m, I, D, incl, decl)
        tfa_n = tfa_n + tfa_cada
    return tfa_n


@jit(nopython=True)
def caculation_anomaly(X, Y, Z, I, D, pop):
    anomaly = List()
    # n_dip = len(pop[0])-1

    for i in range(len(pop)):
        incl = pop[i][len(pop[0]) - 1, 0]
        decl = pop[i][len(pop[0]) - 1, 1]
        m = pop[i][len(pop[0]) - 1, 2]
        spheres = pop[i][0:len(pop[0]) - 1, :]
        anomaly.append(tfa_n_dips(incl, decl, m, len(pop[0]) - 1, X, Y, Z, I, D, spheres))
    return anomaly


def caculation_onlyone_anomaly(X, Y, Z, I, D, pop, m):

    spheres = []
    incl = pop[len(pop[0]) - 1, 0]
    decl = pop[len(pop[0]) - 1, 1]
    mag = pop[len(pop[0]) - 1, 2]
    for j in range(len(pop[0])-1):
        spheres.append((pop[j][0], pop[j][1], pop[j][2], raio))
    anomaly = (tfa_n_dips(incl, decl, m, len(pop[0])-1, X, Y, Z, I, D, spheres))

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


@jit(nopython=True)
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


@jit(nopython=True)
def relative_error(v_referencia, v_calculado):
#REE = ( | estimado - verdadeiro: / verdadeiro ) * 100%
    ree = (np.abs(v_calculado - v_referencia)/(v_referencia))*100
    return ree


@jit(nopython=True)
def normalize(function):
    min_f, max_f = min(function), max(function)
    n_f = List()
    for i in range(len(function)):
        normal_f = (function[i] - min_f)/(max_f + min_f)
        n_f.append(normal_f)
    return n_f

def mean_z(ind):
    pop_len = len(ind)
    #print(ind)
    z_ = np.zeros((pop_len-1, 1))
    for i in range(pop_len-1):
        z_[i] = ind[i,2]
    mean_z = np.mean(z_)
    return mean_z


@jit(nopython=True)
def deversity_calculate(populacao, param):
    pop_l = len(populacao)
    ind_l = len(populacao[0])
    if param == 'X':
        m_ = np.zeros((ind_l-1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][0:ind_l-1,0]
    
    if param == 'Y':
        m_ = np.zeros((ind_l-1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][0:ind_l-1,1]
    
    if param == 'Z':
        m_ = np.zeros((ind_l-1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][0:ind_l-1,2]
    
    if param == 'incl':
        m_ = np.zeros((1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][ind_l-1,0]
    
    if param == 'decl':
        m_ = np.zeros((1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][ind_l-1,1]
    
    if param == 'mom':
        m_ = np.zeros((1, pop_l))
        for i in range(pop_l):
            m_[:, i] = populacao[i][ind_l-1,2]
    
    mean_ = np.mean(m_)
    std_ = np.std(m_)
    
    return mean_, std_

