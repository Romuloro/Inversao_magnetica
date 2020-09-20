import numpy as np
import random
import pandas as pd
import sys
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random


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
        n_par = 6
        for j in range(n_pop):
            individuo = np.zeros((n_dip, n_par))
            coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n_dip)
            incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n_dip, homogeneo)
            for i in range(n_dip):
                individuo[i][0], individuo[i][1], individuo[i][2], individuo[i][3], individuo[i][4], individuo[i][5] = coodX[i], coodY[i], coodZ[i], incl[i], decl[i], mag[i]
            pop.append(individuo)
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
    raio = 100.0
    fit_cada = []

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

        tfa_dip = sample_random.tfa_n_dips(incl, decl, mag, len(pop[0]), X, Y, Z, I, D, spheres)
        fit_cada.append(sample_random.f_difference(tfa_n_dip, tfa_dip))
    return fit_cada


def tournament_selection(pop, fit_cada):
    """
    Função com o objetivo de selecionar os futuros pais, pelo dinâmica do Torneio.

    :param pop: População com n indivíduos.
    :param fit_cada: O valor de fitness para cada n indivpiduos.

    :return chosen: Lista com os n pais.
    """

    pop_1 = pop.copy()
    chosen = []
    for i in range(int(0.2 * len(pop))):
        capture_select = []
        # ---------------------------- Escolhidos para o torneio ---------------------------------#
        index_select = list(random.sample(range(0, len(pop_1)), k=(int(0.2 * len(pop)))))
        for j in range(int(0.2 * len(pop))):
            capture = [fit_cada[index_select[j]], index_select[j]]
            capture_select.append(capture)
        # ---------------------------- Vencedor do torneio ---------------------------------#
        escolhido = pop_1[min(capture_select[:])[1]]
        # ------------------ Retirada do vencedor da população artificial ------------------------#
        del (pop_1[min(capture_select[:])[1]])
        # ---------------------------- Vencedores do torneio ---------------------------------#
        chosen.append(escolhido)

    return chosen


def crossover(pais_torneio):
    filhos = []
    n_filhos = int(len(pais_torneio) / 2)
    pai = np.array(pais_torneio[0:n_filhos])
    mae = np.array(pais_torneio[n_filhos:len(pais_torneio)])
    prob_pai = random.random()
    prob_mae = random.random()
    den = prob_mae + prob_pai

    for j in range(n_filhos):
        num = (prob_pai * pai[j] + prob_mae * mae[j])
        filho = num / den # Verificar se os n filhos estão dentro dos limites de busca.
        filhos.append(filho)

    return filhos


def mutacao(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo):

    prob_mut = 0.01
    for index, rand_mut in enumerate(filho):
        rand_mut = random.random()
        if prob_mut > rand_mut:
            dip_select = random.randint(0, (len(filho[0]) - 1)) #Seleção qual dipolo será mutado.
            param_select = random.randint(0, (len(filho[0][0]) - 1)) #Selecão qual parâmetro será mutado.
            if param_select <= 2:
                coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                if param_select == 0:
                    filho[index][dip_select][param_select] = float(coodX[0])
                elif param_select == 1:
                    filho[index][dip_select][param_select] = float(coodY[0])
                elif param_select == 2:
                    filho[index][dip_select][param_select] = float(coodZ[0])
            else:
                incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n,
                                                                  homogeneo)
                if param_select == 3:
                    filho[index][dip_select][param_select] = float(incl[0])
                elif param_select == 4:
                    filho[index][dip_select][param_select] = float(decl[0])
                elif param_select == 5:
                    filho[index][dip_select][param_select] = float(mag[0])

    return filho

def mutacao_vhomo(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo):

    prob_mut = 0.01
    for index, rand_mut in enumerate(filho):
        rand_mut = random.random()
        if prob_mut > rand_mut:
            dip_select = random.randint(0, (len(filho[0]) - 1)) #Seleção qual dipolo será mutado.
            param_select = random.randint(0, (len(filho[0][0]) - 1)) #Selecão qual parâmetro será mutado.
            if param_select <= 2:
                coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                if param_select == 0:
                    filho[index][dip_select][param_select] = float(coodX[0])
                elif param_select == 1:
                    filho[index][dip_select][param_select] = float(coodY[0])
                elif param_select == 2:
                    filho[index][dip_select][param_select] = float(coodZ[0])
            else:
                incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n,
                                                                  homogeneo)
                if homogeneo == True:
                    if param_select == 3:
                        filho[index][:][param_select] = float(incl[0])
                    elif param_select == 4:
                        filho[index][:][param_select] = float(decl[0])
                    elif param_select == 5:
                        filho[index][:][param_select] = float(mag[0])
                else:
                    if param_select == 3:
                        filho[index][dip_select][param_select] = float(incl[0])
                    elif param_select == 4:
                        filho[index][dip_select][param_select] = float(decl[0])
                    elif param_select == 5:
                        filho[index][dip_select][param_select] = float(mag[0])

    return filho

def elitismo(pop, filhos, fit_cada):
    n_fica = (len(pop) - len(filhos))
    df = pd.DataFrame(fit_cada)
    x = df.sort_values(0, ascending=True)
    piores = x.index[n_fica:]
    for index, pos in enumerate(piores):
        pop[pos] = filhos[index]

    return pop
