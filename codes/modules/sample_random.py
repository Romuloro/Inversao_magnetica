import numpy as np
import sys
a = sys.path.append('../modules/') # endereco das funcoes implementadas por voce!
import sphere

def f_difference (dado_referencia, dado_calculado):
    """
    Função com o objetivo de calcular o valor da função diferença entre os dados de referência para os dados calculados.

    :param dado_referencia: O dado de referência.
    :param dado_calculado: O dado calculado que será comparado.
    :return: Valor da função diferença.
    """

    std = np.std(dado_referencia)
    dif = (dado_referencia - dado_calculado)**2 / (std**2)
    rms = np.sum(dif) / len(dif)
    
    return rms


def sample_random_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n):
    """
    Função com o objetivo de criar de forma randomica as coordenadas para n corpos.

    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: sample_random_coordinated(**dicionario).

    :param dicionario: xmax - O valor máximo da coordenada X.
                       ymax - O valor máximo da coordenada Y.
                       zlim - O valor máximo da coordenada Z.
                       xmin - O valor minímo da coordenada X.
                       ymin - O valor minímo da coordenada Y.
                       z_min - O valor minímo da coordenada Z.
                       n - número de bolinhas desejadas.
    :return: resultadox - Lista com o resultado final para das coordenadas no eixo X.
             resultadoy - Lista com o resultado final para das coordenadas no eixo Y.
             resultadoz - Lista com o final para das coordenadas no eixo Z.
    """

    resultadox=[]
    resultadoy=[]
    resultadoz=[]
#---------------------------------------------------------------------------------------------------------------------#
    for i in range(n):
        sorted_x1, sorted_y1,sorted_z1 = (float("{0:.2f}".format(np.random.uniform(xmin, xmax))),
                                      float("{0:.2f}".format(np.random.uniform(ymin, ymax))),
                                      float("{0:.2f}".format(np.random.uniform(z_min, zlim))))
        resultadox.append(sorted_x1)
        resultadoy.append(sorted_y1)
        resultadoz.append(sorted_z1)  
#---------------------------------------------------------------------------------------------------------------------#        
    return resultadox, resultadoy, resultadoz

def sample_random_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo=False):
    """
    Função com o objetivo de criar de forma randomica as propriedades magnéticas para n corpos.

    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: sample_random_mag(**dicionario, homogeneo).

    :param dicionario: inclmax - Valor máximo da inclianção magnética.
                       inclmin = Valor mínimo da inclianção magnética.
                       declmax = Valor máximo da inclianção magnética.
                       declmin = Valor mínimo da declianção magnética.
                       magmax = Valor máximo da magnetização.
                       magmin = Valor mínimo da magnetização.
                       n - número de bolinhas desejadas.
    :param homogeneo: True para valores de magnetização iguais para as n bolinhas.
                      False é a opção default, onde os valores de magnetização é criada de forma randominca.
    :return: incl - Lista com os valores de inclinação magnética.
             decl - Lista com os valores de declinação magnética.
             mag - Lista com os valores de magnetização.
    """

    incl=[]
    decl=[]
    mag=[]
#---------------------------------------------------------------------------------------------------------------------#
    if homogeneo == True:
        sorted_mag =(float("{0:.2f}".format(np.random.uniform(magmax, magmin))))  
#---------------------------------------------------------------------------------------------------------------------#        
        for i in range(n):
            sorted_incl, sorted_decl = (float("{0:.2f}".format(np.random.uniform(inclmax,inclmin))),
                                   float("{0:.2f}".format(np.random.uniform(declmax, declmin))))
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)       
#---------------------------------------------------------------------------------------------------------------------#
    else:
        for i in range(n):
            sorted_incl, sorted_decl, sorted_mag =(float("{0:.2f}".format(np.random.uniform(inclmax,inclmin))),
                                   float("{0:.2f}".format(np.random.uniform(declmax, declmin))),
                                   float("{0:.2f}".format(np.random.uniform(magmax, magmin))))
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)
    
    return incl, decl, mag

def tfa_n_dots(incl, decl, mag, n, Xref, Yref, Zref, I, D, coodX, coodY, coodZ, raio):
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
                       coodX - Lista com os valores de coordenada de cada bolinha no eixo X.
                       coodY - Lista com os valores de coordenada de cada bolinha no eixo Y.
                       coodZ - Lista com os valores de coordenada de cada bolinha no eixo Z.
                       raio - O raio de cada bolinha.
    :return: Uma matrix com os valores de anomália magnética para cada ponto do local estudado.
    """

    sphere1 = []
    n = n
    for i in range(n):
        sphere1.append((coodX[i], coodY[i], coodZ[i], raio))
#---------------------------------------------------------------------------------------------------------------------#   
    tfa_n = 0
    for i in range(n):
        tfa_cada = sphere.sphere_tfa(Xref,Yref,Zref,sphere1[i],mag[i],I,D,incl[i],decl[i])
        tfa_n += tfa_cada
    
    return tfa_n