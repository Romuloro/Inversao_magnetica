import numpy as np
import random
import sys
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random


def create_population(xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n,
                      homogeneo):
    """
    Função com o objetivo de criar uma população com n indivíduos randômicos, que estaram de acordo com os parâmetros
    escolhidos.

    :param xmax: O valor máximo da coordenada X.
    :param ymax: O valor máximo da coordenada Y.
    :param zlim: O valor máximo da coordenada Z.
    :param xmin: O valor minímo da coordenada X.
    :param ymin: O valor minímo da coordenada Y.
    :param z_min: O valor minímo da coordenada Z.
    :param n: número de indivíduos desejados.
    :param inclmax: Valor máximo da inclianção magnética.
    :param inclmin: Valor mínimo da inclianção magnética.
    :param declmax: Valor máximo da inclianção magnética.
    :param declmin: Valor mínimo da declianção magnética.
    :param magmax: Valor máximo da magnetização.
    :param magmin: Valor mínimo da magnetização.
    :param homogeneo: True para valores de magnetização iguais para as n bolinhas.
                      False é a opção default, onde os valores de magnetização é criada de forma randominca.
    """
    coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
    incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo)
    dipolos_pop = []
    # raio = 100.0 # Valor do raio em metros, escolhido!!
    if n < 10:
        print(f'Por favor faça uma população com mais de 10 indivíduos')
    else:
        for i in range(n):
            dipolo = [coodX[i], coodY[i], coodZ[i], incl[i], decl[i], mag[i]]
            dipolos_pop.append(dipolo)

    return dipolos_pop


def fit_value(X, Y, Z, pop, tfa_n_bolinhas):
    """
    Função que calcula o fitness de cada indivíduo da população.

    :param X:
    :param Y:

    """
    list_sphere = []
    list_mag = []
    list_incl = []
    list_decl = []
    fit_cada = []

    for i in range(len(pop)):
        list_sphere.append((pop[i][0], pop[i][1], pop[i][2], 100.0))
        list_mag.append(pop[i][5])
        list_incl.append(pop[i][3])
        list_decl.append((pop[i][4]))

        tfa_cada = sphere.sphere_tfa(X, Y, Z, list_sphere[i], list_mag[i], 30.0, 50.0, list_incl[i], list_decl[i])

        fit = sample_random.f_difference(tfa_n_bolinhas, tfa_cada)
        fit_cada.append(float("{0:.2f}".format(fit)))

    return fit_cada


def tournament_selection(pop, fit_cada):
    pop_1 = pop.copy()
    chosen = []
    capture_select = []
    if int(0.2 * len(pop)) < 2:
        print(f'Por favor faça uma população com mais de 10 indivíduos')
    else:
        for i in range(int(0.2 * len(pop))):
            # ---------------------------- Escolhidos para o torneio ---------------------------------#
            index_select = list(random.sample(range(0, len(pop_1)), k=(int(0.2 * len(pop)))))
            capture = [index_select[i], fit_cada[index_select[i]]]
            capture_select.append(capture)
            # ---------------------------- Vencedor do torneio ---------------------------------#
            escolhido = pop_1[min(capture_select)[0]]
            # ------------------ Retirada do vencedor da população artificial ------------------------#
            del (pop_1[min(capture_select)[0]])
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
        filho = num / den
        filho = list(filho)
        filhos.append(filho)

    return filhos


def mutacao(filho, xmax, xmin, ymax, ymin, zlim, z_min, inclmax, inclmin, declmax, declmin, magmax, magmin, n,
            homogeneo):
    prob_mut = 0.01
    for ind, dipolo in enumerate(filho):
        rand_mut = random.random()
        if prob_mut > rand_mut:
            n_select = random.randint(0, (len(filho) - 1))
            param_select = random.randint(0, (len(filho[0])))
            if param_select <= 2:
                coodX, coodY, coodZ = sample_random.sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n)
                if param_select == 0:
                    filho[n_select][param_select] = float(coodX[0])
                elif param_select == 1:
                    filho[n_select][param_select] = float(coodY[0])
                elif param_select == 2:
                    filho[n_select][param_select] = float(coodZ[0])
            else:
                incl, decl, mag = sample_random.sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n,
                                                                  homogeneo)
                if param_select == 3:
                    filho[n_select][param_select] = float(incl[0])
                elif param_select == 4:
                    filho[n_select][param_select] = float(decl[0])
                elif param_select == 5:
                    filho[n_select][param_select] = float(mag[0])

    return filho


def elitismo(pop, filhos, fit_cada):
    pop_fit = []
    n_fica = (len(pop) - len(filhos))

    for i in range(len(pop)):
        fit_pop = [pop[i][0], pop[i][1], pop[i][2], pop[i][3], pop[i][4], pop[i][5], fit_cada[i]]
        pop_fit.append(fit_pop)
        sort_pop = sorted(pop_fit, key=lambda pop_fit: pop_fit[:][6])

    for i in range(len(sort_pop)):
        del (sort_pop[i][6])

    del (sort_pop[n_fica: len(pop)])
    for i in range(len(filhos)):
        new_individuo = filhos[i]
        sort_pop.append(new_individuo)

    return sort_pop

