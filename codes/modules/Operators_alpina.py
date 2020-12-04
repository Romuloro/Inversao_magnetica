import numpy as np
import random
import pandas as pd
import sys
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random, aux_operators_array, Operators_array



def n_elitismo(pop, filhos, fit_cada):
    n_pop = pop.copy()
    n_fica = int(len(pop) - (len(filhos)-(0.8*len(pop))))
    index_select = list(random.sample(range(0, len(pop)), k=(n_fica)))
    #print('N fica é =', n_fica)
    df = pd.DataFrame(fit_cada)
    #x = df.sort_values(0, ascending=True) #Ordenar os valores de acordo com o menor fit.
    #piores = x.index[:]
    #print(piores)
    #print(len(filhos))
    for index, pos in enumerate(index_select): #Substituir os piores indivíduos pelos filhos
        n_pop[pos] = filhos[index]
        

    return n_pop


def create_population(xmax, xmin, ymax, ymin, zlim, z_min, n_dip, n_pop, homogeneo):
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
        pop = []
        n_par = 2
        for j in range(n_pop):
            cood = np.zeros((n_dip, n_par))
            coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n_dip)
            #incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, 1, homogeneo)
            for i in range(n_dip):
                cood[i][0], cood[i][1] = coodX[i], coodY[i]
            #cood[n_dip][0], cood[n_dip][1], cood[n_dip][2] = incl[0], decl[0], mag[0]
            pop.append(cood)
        return pop
    else:
        return print('Por favor. Coloque o número de indivíduos maior ou igual a 1')



def mutacao_vhomo(filho, xmax, xmin, ymax, ymin, zlim, z_min, n, homogeneo):

    prob_mut = 0.05
    for index, rand_mut in enumerate(filho): #Index = qual será o indivíduo que será mutado.
        rand_mut = random.random()
        if prob_mut > rand_mut:
            param_select = random.randint(0, 1) #Selecão qual parâmetro será mutado.
            coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
            if param_select == 0:
                filho[index][0, param_select] = coodX[0]
            elif param_select == 1:
                filho[index][0, param_select] = coodY[0]

    return filho



def function_alphina(pop):
    f_alp = []
    for i in range(len(pop)):
        f = -(np.sin(pop[i][0,0])*np.sin(pop[i][0,1])*((pop[i][0,0]*pop[i][0,1])**(1/2)))
        f = f.real
        f_alp.append(f)
    return f_alp


def function_alphina_normal(X, Y):
    f = -(np.sin(X)*np.sin(Y)*((X * Y)**(1/2)))
    f = f.real
    return f

def single_p_crossover(pop_inicial, X = 1):
    filhos = []
    n_filhos = int(len(pop_inicial) / 2)
    pai = np.array(pop_inicial[0:n_filhos])
    inv_pai = pai[::-1]
    mae = np.array(pop_inicial[n_filhos:len(pop_inicial)])
    for i in range(n_filhos):
        pai_new = np.zeros((1,2))
        mae_new = np.zeros((1,2))
        invpai_new = np.zeros((1,2))
        pai_new[0,0], pai_new[0,1] = pai[i][0,:X], mae[i][0,X:]
        mae_new[0,0], mae_new[0,1] = mae[i][0,:X], pai[i][0,X:]
        invpai_new[0,0], invpai_new[0,1] = inv_pai[i][0,:X], mae[i][0,X:]
        filhos +=[pai_new, mae_new, invpai_new]
    return filhos


def ag(populacao, **filhos_mut):
    ind_better = []
    fit_ = function_alphina(populacao)
    min_fit = fit_.index(min(fit_))
    ind_better.append(populacao[min_fit])
    val_fit = min(fit_)
    pais_, escolhidos = Operators_array.tournament_selection(populacao, fit_)
    filho_ = Operators_array.crossover_polyamory(pais_, escolhidos, fit_) #single_p_crossover(pais_)
    #print(len(filho_))
    filhos_ = mutacao_vhomo(filho_, **filhos_mut)
    nova_populacao = Operators_array.elitismo(populacao, filhos_, fit_)
    print(nova_populacao)
    print(len(nova_populacao))
    return nova_populacao, val_fit
