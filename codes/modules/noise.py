#------------------------------------------------------------------------------------
import numpy as np
import random
#------------------------------------------------------------------------------------
def noise_gaussiana(t, mu, sigma, v):
    '''
    Função para a criação de um ruído gaussiano.
    
    Inputs: 
    t = tamanho do vetor onde será aplicado o ruído.
    mu = média.
    sigma = desvio padrão.
    v = Matriz onde o ruido será aplicado.
    
    Output:
    noise_gaussiana = Matriz contendo ruído
    '''
    
    noise = np.random.normal(mu, sigma, t)
    noise_gaussiana = v + noise
    
    return noise_gaussiana

