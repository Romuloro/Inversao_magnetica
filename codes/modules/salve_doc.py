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
    pastac.extend('/data_mag.cvs')
    pasta3 = ''.join(pastac)
    matriz.to_csv(pasta3, index = False, header = True)



def create_diretorio_dipolos(dicionario, o_ind, i_ind, f_ind):
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
    endereco = './Testes_congresso/'
    pasta.extend([endereco, data_e_hora])
    pastac = pasta.copy()
    pastac2 = pasta.copy()
    pastac3 = pasta.copy()
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
    #Exportar o cvs com os resultados
    pastac.extend('/frist_ind.cvs')
    pasta3 = ''.join(pastac)
    o_ind.to_csv(pasta3, index = False, header = False)
    
    pastac2.extend('/intermediate_ind.cvs')
    pasta4 = ''.join(pastac2)
    i_ind.to_csv(pasta4, index = False, header = False)
    
    pastac3.extend('/final_ind.cvs')
    pasta5 = ''.join(pastac3)
    f_ind.to_csv(pasta5, index = False, header = False)



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