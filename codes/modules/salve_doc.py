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


def reshape_matrix(dicionario):
    '''
    Criação de data frame para a organização dos dados.
    
    dicionario = dicionário com todos as entradas organizadas da seguinte forma.
    dicionario = {'nx': número de observações no eixo X,
                  'ny': número de observações no eixo Y,
                  'X': observações no eixo X,
                  'Y': observações no eixo Y,
                  'Z': altura de voo,
                  'ACTn': volares da anomalia magnética calculada do corpo
                  }
    '''
    
    #Elementos da matriz:
    X = dicionario.get('X')
    Y = dicionario.get('Y')
    Z = dicionario.get('Z')
    ACTn = dicionario.get('ACTn')
    ny = dicionario.get('ny')
    nx = dicionario.get('nx')
    #-----------------------------------
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