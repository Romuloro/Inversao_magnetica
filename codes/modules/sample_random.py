import numpy as np
import sys
a = sys.path.append('../modules/') # endereco das funcoes implementadas por voce!
import sphere
from numba import jit
from numba.typed import List


@jit(nopython=True)
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

    resultadox= List()
    resultadoy= List()
    resultadoz= List()
#---------------------------------------------------------------------------------------------------------------------#
    for i in range(n):
        sorted_x1, sorted_y1,sorted_z1 = (np.random.uniform(xmin, xmax),
                                      np.random.uniform(ymin, ymax),
                                      np.random.uniform(z_min, zlim))
        resultadox.append(sorted_x1)
        resultadoy.append(sorted_y1)
        resultadoz.append(sorted_z1)  
#---------------------------------------------------------------------------------------------------------------------#        
    return resultadox, resultadoy, resultadoz

@jit(nopython=True)
def sample_random_normal_coordinated(xmax, xmin, ymax, ymin, zlim, z_min, n):
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

    resultadox= List()
    resultadoy= List()
    resultadoz= List()
    mu_x, sigma_x = 0, xmax/4.4
    mu_y, sigma_y = 0, ymax/4.4
    mu_z = (zlim + z_min)/2
    sigma_z = round(((abs(zlim) - mu_z)/4.4),2)
#---------------------------------------------------------------------------------------------------------------------#
    for i in range(n):
        sorted_x1, sorted_y1,sorted_z1 = (np.random.normal(mu_x, sigma_x),
                                      np.random.normal(mu_y, sigma_y),
                                      np.random.normal(mu_z, sigma_z))
        resultadox.append(sorted_x1)
        resultadoy.append(sorted_y1)
        resultadoz.append(sorted_z1)  
#---------------------------------------------------------------------------------------------------------------------#        
    return resultadox, resultadoy, resultadoz


@jit(nopython=True)
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
    :param homogeneo: True para valores de inclinação, declinação e magnetização iguais para as n bolinhas.
                      False é a opção default, onde os valores de inclinação, declinação e magnetização é criada de forma randômica.
    :return: incl - Lista com os valores de inclinação magnética.
             decl - Lista com os valores de declinação magnética.
             mag - Lista com os valores de magnetização.
    """
    porc = 0.2
    incl= List()
    decl= List()
    mag= List()
    magmax = magmax #+ porc*magmax
    magmin = magmin #- porc*magmin
    #print(magmax, magmin)
#---------------------------------------------------------------------------------------------------------------------#
    if homogeneo == True:
        sorted_incl, sorted_decl, sorted_mag =(np.random.uniform(inclmax,inclmin),
                                   np.random.uniform(declmax, declmin),
                                   np.random.uniform(magmax, magmin))  
#---------------------------------------------------------------------------------------------------------------------#        
        for i in range(n):
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)       
#---------------------------------------------------------------------------------------------------------------------#
    else:
        for i in range(n):
            sorted_incl, sorted_decl, sorted_mag =(np.random.uniform(inclmax,inclmin),
                                   np.random.uniform(declmax, declmin),
                                   np.random.uniform(magmax, magmin))
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)
    
    return incl, decl, mag

@jit(nopython=True)
def sample_random_normal_mag(inclmax, inclmin, declmax, declmin, magmax, magmin, n, homogeneo=False):
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
    :param homogeneo: True para valores de inclinação, declinação e magnetização iguais para as n bolinhas.
                      False é a opção default, onde os valores de inclinação, declinação e magnetização é criada de forma randômica.
    :return: incl - Lista com os valores de inclinação magnética.
             decl - Lista com os valores de declinação magnética.
             mag - Lista com os valores de magnetização.
    """
    porc = 0.2
    incl= List()
    decl= List()
    mag= List()
    magmax = magmax #+ porc*magmax
    magmin = magmin #- porc*magmin
    #print(magmax, magmin)
    mu_incl = (inclmax + inclmin)/2
    sigma_incl = round(((abs(inclmax) - abs(mu_incl))/4.4),2)
    mu_decl = (declmax + declmin)/2
    sigma_decl = round(((abs(declmax) - abs(mu_decl))/4.4),2)
    mu_mag = (magmax + magmin)/2
    sigma_mag = round(((abs(magmax) - abs(mu_mag))/4.4),2)
#---------------------------------------------------------------------------------------------------------------------#
    if homogeneo == True:
        sorted_incl, sorted_decl, sorted_mag =(np.random.normal(mu_incl, sigma_incl),
                                   np.random.normal(mu_decl, sigma_decl),
                                   np.random.normal(mu_mag, sigma_mag))  
#---------------------------------------------------------------------------------------------------------------------#        
        for i in range(n):
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)       
#---------------------------------------------------------------------------------------------------------------------#
    else:
        for i in range(n):
            sorted_incl, sorted_decl, sorted_mag =(np.random.uniform(inclmax,inclmin),
                                   np.random.uniform(declmax, declmin),
                                   np.random.uniform(magmax, magmin))
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

    sphere1 =  List()
    for i in range(n):
        sphere1.append((coodX[i], coodY[i], coodZ[i], raio))
#---------------------------------------------------------------------------------------------------------------------#   
    tfa_n = 0
    for i in range(n):
        tfa_cada = sphere.sphere_tfa(Xref,Yref,Zref,sphere1[i],mag[i],I,D,incl[i],decl[i])
        tfa_n += tfa_cada
    
    return tfa_n

