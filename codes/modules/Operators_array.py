# --------------------------------------------------------------------------------------------------
# Title: Mag Codes
# Author: Rômulo Rodrigues de Oliveira
# Description: Source codes
# Collaboratores: Rodrigo Bijani
# -----------------------------------------------------------------------------------------

import numpy as np
import random
import pandas as pd
import sys
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random, aux_operators_array

def create_population(xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n_dip, n_pop, homogeneo):
    """
    Função com o objetivo de criar uma população com n indivíduos randômicos, que estaram de acordo com os parâmetros
    escolhidos.

    :param xmax: O valor máximo da coordenada X.
    :param ymax: O valor máximo da coordenada Y.
    :param zlim: O valor máximo da coordenada Z.
    :param xmin: O valor minímo da coordenada X.
    :param ymin: O valor minímo da coordenada Y.
    :param z_min: O valor minímo da coordenada Z.
    :param n_pop: O número de indivíduos desejados na população.
    :param n_dip: O número de dipolos desejados para cada indivíduo.
    :param inclmax: Valor máximo da inclianção magnética.
    :param inclmin: Valor mínimo da inclianção magnética.
    :param declmax: Valor máximo da inclianção magnética.
    :param declmin: Valor mínimo da declianção magnética.
    :param magmax: Valor máximo da magnetização.
    :param magmin: Valor mínimo da magnetização.
    :param homogeneo: True para valores de inclinação, declinação e magnetização iguais para as n dipolos.
                      False é a opção default, onde os valores de inclinação, declinação e magnetização é criada de
                      forma randômica.

    :return pop: Lista com n indivíduos/dipolos criados de forma randômica.
    """
    if n_pop >= 10:
        pop = []
        n_par = 3
        for j in range(n_pop):
            cood = np.zeros((n_dip+1, n_par))
            coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n_dip)
            incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, 1, homogeneo)
            for i in range(n_dip):
                cood[i][0], cood[i][1], cood[i][2] = coodX[i], coodY[i], coodZ[i]
            cood[n_dip][0], cood[n_dip][1], cood[n_dip][2] = incl[0], decl[0], mag[0]
            pop.append(cood)
        return pop
    else:
        return print('Por favor. Coloque o número de indivíduos maior ou igual a 10')


def fit_value(X, Y, Z, I, D, pop, tfa_n_dip):
    """
    Função que calcula o fitness de cada indivíduo da população.

    :param X: Pontos de observação na coordenadas X.
    :param Y: Pontos de observação na coordenadas Y.
    :param Z: Pontos de observação na coordenadas Z.
    :param I: Inclinação magnética regional.
    :param D: Declinação magnética regional.
    :param pop: População com n indivíduos.
    :param tfa_n_dip: Anomalia magnética referência.

    :return fit_cada: Lista com o valor de fitness de cara indivíduo da população.
    """
    fit_cada = []
    anomalia = aux_operators_array.caculation_anomaly(X, Y, Z, I, D, pop) #Cálculo da anomalia
    for i in range(len(pop)):
        fit_cada.append(aux_operators_array.f_difference(tfa_n_dip, anomalia[i])) #Cálculo do fit
    return fit_cada, anomalia


def tournament_selection(pop, fit_cada, p_pop = 0.25):
    """
    Função com o objetivo de selecionar os futuros pais, pelo dinâmica do Torneio.

    :param pop: População com n indivíduos.
    :param fit_cada: O valor de fitness para cada n indivpiduos.

    :return chosen: Lista com os n pais.
    """

    pop_1 = pop.copy()
    chosen = []
    select = []
    for i in range(int(p_pop * len(pop))):
        capture_select = []
        # ---------------------------- Escolhidos para o torneio ---------------------------------#
        index_select = list(random.sample(range(0, len(pop_1)), k=(int(0.2 * len(pop)))))
        for j in range(int(0.2 * len(pop))):
            capture = [fit_cada[index_select[j]], index_select[j]]
            capture_select.append(capture)
        # ---------------------------- Vencedor do torneio ---------------------------------#
        escolhido = pop_1[min(capture_select[:])[1]]
        select.append(min(capture_select[:])[1])
        # ------------------ Retirada do vencedor da população artificial ------------------------#
        del (pop_1[min(capture_select[:])[1]])
        # ---------------------------- Vencedores do torneio ---------------------------------#
        chosen.append(escolhido)

    return chosen, select


def crossover_elitista(pais_torneio, escolhidos, fit):
    filhos = []
    n_filhos = int(len(pais_torneio) / 2)
    pai = np.array(pais_torneio[0:n_filhos])
    mae = np.array(pais_torneio[n_filhos:len(pais_torneio)])
    # Sorteio das probabilidades de forma randômica.
    #prob_pai, prob_mae, den = aux_operators_array.definition_prob(pais_torneio, escolhidos, fit, n_filhos)
    prob_pai = random.random()
    prob_mae = random.random()
    den = prob_mae + prob_pai

    for j in range(n_filhos):
        num = (prob_pai * pai[j] + prob_mae * mae[j])
        filho = num / den # Cálculo do filho
        print('Filho=', j, filho)
        filhos.append(filho)

    return filhos


def mutacao_vhomo(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo):

    prob_mut = 0.05
    for index, rand_mut in enumerate(filho): #Index = qual será o indivíduo que será mutado.
        rand_mut = random.random()
        if prob_mut > rand_mut:
            dip_select = random.randint(0, (len(filho[0]) - 2)) #Seleção qual dipolo será mutado.
            param_select = random.randint(0, (len(filho[0][0]) + 3)) #Selecão qual parâmetro será mutado.
            if param_select <= 2:
                coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                if param_select == 0:
                    filho[index][dip_select][param_select] = coodX[0]
                elif param_select == 1:
                    filho[index][dip_select][param_select] = coodY[0]
                elif param_select == 2:
                    filho[index][dip_select][param_select] = coodZ[0]
            else:
                incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo)
                if param_select == 3:
                    filho[index][len(filho[0])-1][0] = incl[0]
                elif param_select == 4:
                    filho[index][len(filho[0])-1][1] = decl[0]
                elif param_select == 5:
                    filho[index][len(filho[0])-1][2] = mag[0]

    return filho


def elitismo(pop, filhos, fit_cada):
    n_pop = pop.copy()
    n_fica = int(len(pop) - (len(filhos)-(0.8*len(pop))))
    #print('N fica é =', n_fica)
    df = pd.DataFrame(fit_cada)
    x = df.sort_values(0, ascending=True) #Ordenar os valores de acordo com o menor fit.
    piores = x.index[n_fica:]
    for index, pos in enumerate(piores): #Substituir os piores indivíduos pelos filhos
        n_pop[pos] = filhos[index]

    return n_pop


def crossover_polyamory(pais_torneio, escolhidos, fit):
    filhos = []
    n_filhos = int(len(pais_torneio) / 2)
    pai = np.array(pais_torneio[0:n_filhos])
    inv_pai = pai[::-1]
    mae = np.array(pais_torneio[n_filhos:len(pais_torneio)])
    # Sorteio das probabilidades de forma randômica.
    #prob_pai, prob_mae, den = aux_operators_array.definition_prob(pais_torneio, escolhidos, fit, n_filhos)
    probs = np.random.rand(2,3)

    for j in range(n_filhos):
        num0 = (probs[0,0] * pai[j] + probs[0,1] * mae[j])
        filho0 = num0 / (probs[0,0] + probs[0,1]) # Cálculo do filho
        #print('Filho=', j, filho)
        num1 = (probs[1,0] * pai[j] + probs[1,1] * mae[j])
        filho1 = num1 / (probs[1,0] + probs[1,1]) # Cálculo do filho
        
        num2 = (probs[0,0] * inv_pai[j] + probs[0,1] * mae[j])
        filho2 = num2 / (probs[0,0] + probs[0,1])
        
        num3 = (probs[1,0] * inv_pai[j] + probs[1,1] * mae[j])
        filho3 = num3 / (probs[1,0] + probs[1,1])
        
        num4 = (probs[0,2] * inv_pai[j] + probs[1,2] * mae[j])
        filho4 = num4 / (probs[0,2] + probs[1,2])
        
        num5 = (probs[0,1] * inv_pai[j] + probs[1,2] * mae[j])
        filho5 = num5 / (probs[0,1] + probs[1,2])
        
        num6 = (probs[1,1] * inv_pai[j] + probs[0,2] * mae[j])
        filho6 = num6 / (probs[1,1] + probs[0,2])
        
        num7 = (probs[0,0] * inv_pai[j] + probs[1,2] * mae[j])
        filho7 = num7 / (probs[0,0] + probs[1,2])
        
        num8 = (probs[1,1] * pai[j] + probs[1,2] * mae[j])
        filho8 = num8 / (probs[1,1] + probs[1,2])
        
        num9 = (probs[0,2] * pai[j] + probs[1,2] * mae[j])
        filho9 = num9 / (probs[0,2] + probs[1,2])
        
        filhos += [filho0, filho1, filho2, filho3, filho4, filho5, filho6, filho7, filho8, filho9]

    return filhos


def crossover_mix_doubles(pais_torneio, escolhidos, fit):
    filhos = []
    n_filhos = int(len(pais_torneio) / 2)
    n1_4_filhos = int(len(pais_torneio)/4)
    pai = np.array(pais_torneio[0:n_filhos])
    inv_pai = pai[::-1]
    mae = np.array(pais_torneio[n_filhos:len(pais_torneio)])
    m_pai = np.array(pais_torneio[n1_4_filhos:n_filhos])
    m_mae = np.array(pais_torneio[n_filhos:(n_filhos+n1_4_filhos)])
    m_pai = np.concatenate((m_pai, m_mae), axis=0)
    # Sorteio das probabilidades de forma randômica.
    probs = np.random.rand(4)

    for j in range(n_filhos):
        num0 = (probs[0] * pai[j] + probs[1] * mae[j])
        filho0 = num0 / (probs[0] + probs[1]) # Cálculo do filho
        #print('Filho=', j, filho)
        num1 = (probs[2] * pai[j] + probs[3] * mae[j])
        filho1 = num1 / (probs[2] + probs[3]) # Cálculo do filho
        
        num2 = (probs[0] * inv_pai[j] + probs[1] * mae[j])
        filho2 = num2 / (probs[0] + probs[1])
        
        num3 = (probs[2] * inv_pai[j] + probs[3] * mae[j])
        filho3 = num3 / (probs[2] + probs[3])
        
        num4 = (probs[0] * pai[j] + probs[1] * m_pai[j])
        filho4 = num4 / (probs[0] + probs[1])
        
        num5 = (probs[2] * pai[j] + probs[3] * m_pai[j])
        filho5 = num5 / (probs[2] + probs[3])
        
        num6 = (probs[0] * mae[j] + probs[1] * m_pai[j])
        filho6 = num6 / (probs[0] + probs[1])
        
        num7 = (probs[2] * mae[j] + probs[3] * m_pai[j])
        filho7 = num7 / (probs[2] + probs[3])
        
        
        filhos += [filho0, filho1, filho2, filho3, filho4, filho5, filho6, filho7]

    return filhos