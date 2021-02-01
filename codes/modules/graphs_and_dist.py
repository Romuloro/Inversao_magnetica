import numpy as np
import random
import pandas as pd
import sys
import networkx as nx
import math as mt
a = sys.path.append('../modules/')  # endereco das funcoes implementadas por voce!
import sphere, sample_random, aux_operators_array, Operators_array, plot_3D


def create_graph_dipolo(pop_inicial):
    x = pop_inicial[0:len(pop_inicial)-1,0]
    y = pop_inicial[0:len(pop_inicial)-1,1]
    z = pop_inicial[0:len(pop_inicial)-1,2]
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    grafos = nx.Graph()
    for i in range(len(x)):
        grafos.add_node(i ,pos=(x[i],y[i],z[i]))
        for j in range(len(x)):
            grafos.add_edge(i, j, weight=dist_euclidiana(x,y,z)[i][j])

    #Cálculo do MST
    TSG = nx.minimum_spanning_tree( grafos , algorithm='kruskal' )
    return TSG


def dist_euclidiana(x_coord,y_coord, z_coord):
    '''
    This function takes two vectors, x_coord and y_coord, and returns a matrix were the element in the ij position is the distance (considering the euclidian norm, ou l2 norm) betwen the point (x_coord[i],y_coord[i]) and (x_coord[j],y_coord[j]).
    
    Inputs:
    x_coord - numpy array 
    y_coord - numpy array
    
    Output:
    dl1 - numpy array - Matrix of distances
    '''
    
    #Stablishing the error condition
    tamx = np.shape(x_coord)[0]           
    tamy = np.shape(y_coord)[0]
    tamz = np.shape(z_coord)[0]
    if tamx != tamy or tamy != tamz or tamx != tamz:
        raise ValueError("All inputs must have same length!")
        
    #Calculating and savingn the distances
    #tam = ( math.factorial(tamx) )/( 2*(math.factorial(tamx-2)) ) #2 choises over 'tamx'(or 'tamy') posibilities
    distl2_matrix = np.zeros((tamx,tamy))
                                    
    for i in range(tamx):
        for j in range(tamx):
            distl2_matrix[i][j] = ( (x_coord[i] - x_coord[j])**2 + (y_coord[i] - y_coord[j])**2 + (z_coord[i] - z_coord[j])**2 )**(1/2)
    
    return distl2_matrix



def theta_var(MST):
    dm1 = []
    for (u, v, wt) in MST.edges.data('weight'):
        dm1.append(wt)
    dm3 = np.array(dm1)
    variancia = np.var(dm3)
    
    return variancia



def theta_value(pop_inicial):
    theta = []
    MST = []
    #anomalia = aux_operators_array.caculation_anomaly(X, Y, Z, I, D, pop_inicial)  # Cálculo da anomalia
    for i in range(len(pop_inicial)):
        dipolo=pop_inicial[i]
        MST.append(create_graph_dipolo(dipolo))
        theta.append(theta_var(MST[i]))
        dipolo = 0.0
    
    return theta, MST

