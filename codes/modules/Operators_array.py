# --------------------------------------------------------------------------------------------------
# Title: Mag Codes
# Author: Rômulo Rodrigues de Oliveira
# Description: Source codes
# Collaboratores: Rodrigo Bijani
# -----------------------------------------------------------------------------------------

from numba import jit
from numba.typed import List
import numpy as np
import random
import pandas as pd
import sys
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random, aux_operators_array, graphs_and_dist


@jit(nopython=True)
def create_population(xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, mmax, mmin, n_dip, n_pop, homogeneo):
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
    if n_pop >= 1:
        pop = List()
        n_par = 3
        for j in range(n_pop):
            cood = np.zeros((n_dip+1, n_par))
            coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n_dip)
            incl, decl, m = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, mmax, mmin, 1, homogeneo)
            for i in range(n_dip):
                cood[i][0], cood[i][1], cood[i][2] = coodX[i], coodY[i], coodZ[i]
            cood[n_dip][0], cood[n_dip][1], cood[n_dip][2] = incl[0], decl[0], m[0]
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
    fit_cada = List()
    anomalia = aux_operators_array.caculation_anomaly(X, Y, Z, I, D, pop) #Cálculo da anomalia
    for i in range(len(pop)):
        fit_cada.append(aux_operators_array.f_difference(tfa_n_dip, anomalia[i])) #Cálculo do fit
    return fit_cada, anomalia


def tournament_selection(pop, fit_cada, p_pop = 0.3, n_pai = 0.3):
    """
    Função com o objetivo de selecionar os futuros pais, pelo dinâmica do Torneio.

    :param pop: População com n indivíduos.
    :param fit_cada: O valor de fitness para cada n indivpiduos.

    :return chosen: Lista com os n pais.
    """

    pop_1 = pop.copy()
    chosen = []
    select = []
    k=(int(n_pai * len(pop)))
    #print(k)
    for i in range(k):#(int(p_pop * len(pop))):
        if i == k-1:
            capture_select = []
        # ---------------------------- Escolhidos para o torneio ---------------------------------#
            index_select = list(random.sample(range(0, len(pop_1)), k=(int(p_pop * len(pop_1)))))
            for j in range(int(p_pop * len(pop_1))):
                capture = [fit_cada[index_select[j]], index_select[j]]
                capture_select.append(capture)
        # ---------------------------- Vencedor do torneio ---------------------------------#
            escolhido = pop_1[min(capture_select[:])[1]]
            select.append(max(capture_select[:])[1])
        # ------------------ Retirada do vencedor da população artificial ------------------------#
            del (pop_1[max(capture_select[:])[1]])
        # ---------------------------- Vencedores do torneio ---------------------------------#
            chosen.append(escolhido)
 
        else:
            capture_select = []
        # ---------------------------- Escolhidos para o torneio ---------------------------------#
            index_select = list(random.sample(range(0, len(pop_1)), k=(int(p_pop * len(pop_1)))))
            for j in range(int(p_pop * len(pop_1))):
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


'''
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
'''


def mutacao_vhomo(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo):

    prob_mut = 0.15
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



def elitismo(pop, filhos, fit_cada, n_fica=10):
    n_pop = pop.copy()
    #n_fica = 30
    #n_fica = int(len(pop) - (len(filhos)-(0.2*len(pop)))) Colocar o if!!!
    #print('N fica é =', n_fica)
    fit_cada = np.array(fit_cada)
    df = pd.DataFrame(fit_cada)
    x = df.sort_values(0, ascending=True) #Ordenar os valores de acordo com o menor fit.
    piores = x.index[(len(pop) - len(filhos)):]
    for index, pos in enumerate(piores): #Substituir os piores indivíduos pelos filhos
        n_pop[pos] = filhos[index]
    n_pop = List(n_pop)
    return n_pop



def crossover_polyamory(pais_torneio):
    filhos = List()
    n_filhos = int(len(pais_torneio) / 2)
    pai = np.array(pais_torneio[0:n_filhos])
    inv_pai = pai[::-1]
    mae = np.array(pais_torneio[n_filhos:len(pais_torneio)])
    # Sorteio das probabilidades de forma randômica.
    #prob_pai, prob_mae, den = aux_operators_array.definition_prob(pais_torneio, escolhidos, fit, n_filhos)

    for j in range(n_filhos):
        probs = np.random.rand(2, 2)
        num0 = (probs[0,0] * pai[j] + probs[0,1] * mae[j])
        filho0 = num0 / (probs[0,0] + probs[0,1]) # Cálculo do filho
        #print('Filho=', j, filho)
        num1 = (probs[1,0] * inv_pai[j] + probs[1,1] * mae[j])
        filho1 = num1 / (probs[1,0] + probs[1,1]) # Cálculo do filho
        
        '''num2 = (probs[0,0] * inv_pai[j] + probs[0,1] * mae[j])
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
        filho9 = num9 / (probs[0,2] + probs[1,2])'''
        
        filhos += [filho0, filho1 ]#, filho2, filho3, filho4, filho5, filho6, filho7, filho8, filho9]

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


def uniform_crossover(pop_inicial):
    p_shape = tuple(pop_inicial[0].shape) #tuple(mae1.shape)
    filhos = []
    n_filhos = int(len(pop_inicial) / 2)
    pai = pop_inicial[0:n_filhos]
    mae = pop_inicial[n_filhos:len(pop_inicial)]
    for k in range(n_filhos):
        probs = np.random.rand(p_shape[0], p_shape[1])
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
    return filhos



def mutacao_multi_vhomo(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo, prob_mut = 0.05):

    #prob_mut = 0.05
    n_dip = len(filho[0]) - 1
    n_param = 6
    for index, rand_mut in enumerate(filho): #Index = qual será o indivíduo que será mutado.
        rand_mut = random.random()
        if prob_mut > rand_mut:
            dip_select = random.sample(range(0,(len(filho[0]) - 2)), k=(int(n_dip/2))) #Seleção qual dipolo será mutado.
            param_select = random.sample(range(0, (len(filho[0][0]) + 3)), k=(int(n_param/2))) #Selecão qual parâmetro será mutado.
            for round in dip_select:
                for param in param_select:
                    if param <= 2:
                        coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                    if param == 0:
                        filho[index][round][param] = coodX[0]
                    elif param == 1:
                        filho[index][round][param] = coodY[0]
                    elif param == 2:
                        filho[index][round][param] = coodZ[0]
                    else:
                        incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo)
                    if param == 3:
                        filho[index][len(filho[0])-1][0] = incl[0]
                    elif param == 4:
                        filho[index][len(filho[0])-1][1] = decl[0]
                    elif param == 5:
                        filho[index][len(filho[0])-1][2] = mag[0]

    return filho

def mutacao_multi_vhomo_normal(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo, prob_mut = 0.05):

    #prob_mut = 0.05
    n_dip = len(filho[0]) - 1
    n_param = 6
    for index, rand_mut in enumerate(filho): #Index = qual será o indivíduo que será mutado.
        rand_mut = random.random()
        if prob_mut > rand_mut:
            dip_select = random.sample(range(0,(len(filho[0]) - 2)), k=(int(n_dip/2))) #Seleção qual dipolo será mutado.
            param_select = random.sample(range(0, (len(filho[0][0]) + 3)), k=(int(n_param/2))) #Selecão qual parâmetro será mutado.
            for round in dip_select:
                for param in param_select:
                    if param <= 2:
                        coodX, coodY, coodZ = sample_random.sample_random_normal_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                    if param == 0:
                        filho[index][round][param] = coodX[0]
                    elif param == 1:
                        filho[index][round][param] = coodY[0]
                    elif param == 2:
                        filho[index][round][param] = coodZ[0]
                    else:
                        incl, decl, mag = sample_random.sample_random_normal_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo)
                    if param == 3:
                        filho[index][len(filho[0])-1][0] = incl[0]
                    elif param == 4:
                        filho[index][len(filho[0])-1][1] = decl[0]
                    elif param == 5:
                        filho[index][len(filho[0])-1][2] = mag[0]

    return filho


def final_fit(X, Y, Z, I, D, pop, tfa_n_dip, lamb):
    fit_gamma = []
    gamma = []
    fit_, anomaly = fit_value(X, Y, Z, I, D, pop, tfa_n_dip)
    theta, MST = graphs_and_dist.theta_value(pop)
    #shape = shape_anomaly(X, Y, Z, I, D,tfa_n_dip, pop)
    for i in range(len(pop)):
        #final_fit = fit_[i] + lamb * theta[i]
        fit_gamma.append(fit_[i] + (lamb * (theta[i])))
        #gamma.append(fit_[i] + (lamb * (theta[i])))
    
    return fit_gamma, anomaly, MST, theta, fit_



def tournament_selection_ranking_diversit(populacao, fit_, n_regiao= 5):
    n_pop = populacao.copy()
    pais_ = []
    n_len = int(len(populacao)/n_regiao)
    k = 20
    v_regiao_pop = []
    v_regiao_fit = []
    pop_ord = [None]*len(n_pop)
    fit_ord = [None]*len(n_pop)
    fit_ = np.array(fit_)
    # Ordenando os valores da população e do phi
    df = pd.DataFrame(fit_)
    x = df.sort_values(0, ascending=True) #Ordenar os valores de acordo com o menor fit.
    y = x.index[0:len(x)]
    for index, pos in enumerate(y):
        pop_ord[index] = n_pop[pos]
        fit_ord[index] = fit_[pos]
    #Dividindo as regiões
    for i in range(n_regiao):
        v_pega_pop = pop_ord[(i*n_len):(n_len*(i+1))]
        v_pega_fit = fit_ord[(i*n_len):(n_len*(i+1))]
        v_regiao_pop.append(v_pega_pop)
        v_regiao_fit.append(v_pega_fit)
    #Escolhendo os pais via torneios
        aprovados, escolhidos = tournament_selection(v_regiao_pop[i], v_regiao_fit[i])
        pais_.extend(aprovados)
    
    return pais_


def eletismo_constraint(X, Y, Z, I, D, filho, anomaly, populacao, choose):
    normal_gama, gama, anomaly, MST, theta, phi = final_fit(X, Y, Z, I, D, filho, anomaly, lamb = 0.00005)
    fit_cada = np.array(normal_gama)
    df = pd.DataFrame(fit_cada)
    x = df.sort_values(0, ascending=True) #Ordenar os valores de acordo com o menor fit.
    n_select = len(populacao) - len(choose)
    melhores = x.index[0:n_select]
    for i in range(n_select):
        change_index = melhores[i]
        choose.append(filho[melhores[i]])
    
    return choose


def constraint_violation(phi, theta, populacao):
    choose = []
    m_phi = np.mean(phi)
    m_theta = np.mean(theta)
    for i in range(len(phi)):
        if phi[i] < m_phi - 0.6*m_phi and theta[i] < m_theta - 0.6*m_theta:
            i_phi = phi.index(phi[i])
            choose.append(populacao[i_phi])
    
    return choose


def shape_value (dado_referencia, pop):
    sm_1, sm_2, sm = 0.0,0.0,0.0
    sm_1 = dado_referencia * pop
    sh_1 = np.sum(sm_1)
    sm_2 = dado_referencia**2
    sh_2 = np.sum(sm_2)
    
    alf = sh_1/sh_2
    
    sm = np.sum((alf*dado_referencia) - pop)**2
    
    fit_sm = np.sqrt(sm)
    return fit_sm

def shape_anomaly(X, Y, Z, I, D, dado_referencia, pop):
    fit_sm = []
    anomalia = aux_operators_array.caculation_anomaly(X, Y, Z, I, D, pop) #Cálculo da anomalia
    for k in range(len(pop)):
        fit_sm.append(shape_value(dado_referencia, anomalia[k]))
    
    return fit_sm
    