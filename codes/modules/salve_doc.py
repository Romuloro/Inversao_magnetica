import numpy
import os
import pandas as pd
from datetime import datetime


def create_diretorio(dicionario, matriz):
    '''
    Função com a finalidade de criar um diretório no qual serão armazenados os dados de cada parte do processo. Também a criação de um  arquivo .txt com os parametros utilizados e de um documento .csv com os resultados.
    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'Data da Modelagem': data_e_hora,
                  'Tipo de Modelagem': Tipo de Modelagem,
                  'números de corpos': n, - número de corpos modelados
                  'Coordenadas do prisma 1 (x1, x2, y1, y2, z1, z2)': [x1, x2, y1, y2, z1, z2],
                  'Coordenadas do prisma 2 (x3, x4, y3, y4, z3, z4)': [x3, x4, y3, y4, z3, z4],
                  'Coordenadas do prisma n (x5, x6, y5, y6, z5, z6)': [x5, x6, y5, y6, z5, z6],
                  'Mergulho': 'positivo' or 'negativo', - orientação da escada. Positivo(esquerda -> direta); Negativo(direta -> esquerda)
                  'Informação da fonte (Mag, Incl, Decl)': [Mi, inc, dec],
                  'Informação regional (Camp.Geomag.Principal, Incl, Decl)': [Fi, I, D]                  
                  }
    '''
    
    data_e_hora = dicionario.get('Data da Modelagem')
    #----------------------------------------------------------------------------------------------------#
    #Criando a posta
    pasta = []
    endereco = './Logfile/'
    pasta.extend([endereco, data_e_hora])
    pastac = pasta.copy()
    pasta1 = ''.join(pasta)
    os.mkdir(pasta1)
    #----------------------------------------------------------------------------------------------------#
    #Exportar o txt com os parametros
    df = pd.Series(dicionario)
    pasta.extend('/parametros.txt')
    pasta2 = ''.join(pasta)
    df.to_csv(pasta2, sep=' ')
    #----------------------------------------------------------------------------------------------------#
    #Exportar o cvs com os resultados
    pastac.extend('/data_mag.csv')
    pasta3 = ''.join(pastac)
    matriz.to_csv(pasta3, index = False, header = True)



def create_diretorio_dipolos(dicionario, o_ind, i_ind, f_ind, incl, decl, gamma, theta, phi, mom):
    '''
    Função com a finalidade de criar um diretório no qual serão armazenados os dados de cada parte do processo. Também a criação de um  arquivo .txt com os parametros utilizados e de um documento .csv com os resultados.
    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'Data da Modelagem': data_e_hora,
                  'Tipo de Modelagem': Tipo de Modelagem,
                  'números de corpos': n, - número de corpos modelados
                  'Coordenadas do prisma 1 (x1, x2, y1, y2, z1, z2)': [x1, x2, y1, y2, z1, z2],
                  'Coordenadas do prisma 2 (x3, x4, y3, y4, z3, z4)': [x3, x4, y3, y4, z3, z4],
                  'Coordenadas do prisma n (x5, x6, y5, y6, z5, z6)': [x5, x6, y5, y6, z5, z6],
                  'Mergulho': 'positivo' or 'negativo', - orientação da escada. Positivo(esquerda -> direta); Negativo(direta -> esquerda)
                  'Informação da fonte (Mag, Incl, Decl)': [Mi, inc, dec],
                  'Informação regional (Camp.Geomag.Principal, Incl, Decl)': [Fi, I, D]                  
                  }
    '''
    
    data_e_hora = dicionario.get('Data da Modelagem')
    #----------------------------------------------------------------------------------------------------#
    #Criando a posta
    os.chdir('/home/romulo/my_project_dir/Inversao_magnetica/codes/tests/Dissertacao/Test_sintetico/Dique_inclinado')
    pasta = []
    endereco = './'
    pasta.extend([endereco, data_e_hora])
    pastac = pasta.copy()
    pastac2 = pasta.copy()
    pastac3 = pasta.copy()
    pastac4 = pasta.copy()
    pastac5 = pasta.copy()
    pastac6 = pasta.copy()
    pastac7 = pasta.copy()
    pastac8 = pasta.copy()
    pastac9 = pasta.copy()
    pasta1 = ''.join(pasta)
    os.mkdir(pasta1)
    #----------------------------------------------------------------------------------------------------#
    #Exportar o txt com os parametros
    df = pd.Series(dicionario)
    pasta.extend('/parametros.txt')
    pasta2 = ''.join(pasta)
    df.to_csv(pasta2, sep=' ')
    #----------------------------------------------------------------------------------------------------#
    o_ind = pd.DataFrame(data = o_ind)
    i_ind = pd.DataFrame(data = i_ind)
    f_ind = pd.DataFrame(data = f_ind)
    incl = pd.DataFrame(data = incl)
    decl = pd.DataFrame(data = decl)
    mom = pd.DataFrame(data = mom)
    gamma = pd.DataFrame(data = gamma)
    theta = pd.DataFrame(data = theta)
    phi = pd.DataFrame(data = phi)
    #Exportar o cvs com os resultados
    pastac.extend('/frist_ind.csv')
    pasta3 = ''.join(pastac)
    o_ind.to_csv(pasta3, index = False, header = False)
    
    pastac2.extend('/intermediate_ind.csv')
    pasta4 = ''.join(pastac2)
    i_ind.to_csv(pasta4, index = False, header = False)
    
    pastac3.extend('/final_ind.csv')
    pasta5 = ''.join(pastac3)
    f_ind.to_csv(pasta5, index = False, header = False)
    
    pastac4.extend('/incl.csv')
    pasta6 = ''.join(pastac4)
    incl.to_csv(pasta6, index = False, header = False)
    
    pastac5.extend('/decl.csv')
    pasta7 = ''.join(pastac5)
    decl.to_csv(pasta7, index = False, header = False)
    
    pastac6.extend('/gamma.csv')
    pasta8 = ''.join(pastac6)
    gamma.to_csv(pasta8, index = False, header = False)
    
    pastac7.extend('/phi.csv')
    pasta9 = ''.join(pastac7)
    phi.to_csv(pasta9, index = False, header = False)
    
    pastac8.extend('/theta.csv')
    pasta10 = ''.join(pastac8)
    theta.to_csv(pasta10, index = False, header = False)
    
    pastac9.extend('/mom.csv')
    pasta11 = ''.join(pastac9)
    theta.to_csv(pasta10, index = False, header = False)


def reshape_matrix(X, Y, Z, ACTn, nx, ny):
    '''
    Criação de data frame para a organização dos dados.
    As entradas da função é feita da forma clássica ou através de um dicionário que é descompactado.
    O dicinário deve conter as chaves nomeadas de forma identica aos parâmetros de entrada da função.
    Exemplo de entrada: reshape_matrix(**dicionario).

    dicionario = {'nx': número de observações no eixo X,
                  'ny': número de observações no eixo Y,
                  'X': observações no eixo X,
                  'Y': observações no eixo Y,
                  'Z': altura de voo,
                  'ACTn': volares da anomalia magnética calculada do corpo
                  }
    '''

    #Reshape da Matriz:
    linha = nx * ny
    ACTn = numpy.reshape(ACTn, (linha, 1))
    X1 = numpy.reshape(X, (linha, 1))
    Y1 = numpy.reshape(Y, (linha, 1))
    Z1 = numpy.reshape(Z, (linha, 1))
    cabecalho = ['North(m)']
    Data_f = pd.DataFrame(data = X1, index = None, columns=cabecalho)
    Data_f['East(m)'] = Y1
    Data_f['Altura de voo(m)'] = Z1
    Data_f['Anomalia Magnética(nT)'] = ACTn
    
    return Data_f