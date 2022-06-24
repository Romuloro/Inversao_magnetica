import numpy as np
import sys
a = sys.path.append('../modules/') # endereco das funcoes implementadas por voce!
import sphere
from numba import jit
from numba.typed import List

"""
class Source:

    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z

class Individuo_Dipolar(Source):

    def __init__(self,  x, y, z, incl, decl, moment):
        super().__init__(x, y, z)
        self.__incl = incl
        self.__decl = decl
        self.__moment = moment

    def dipole_group(self, x, y, z, incl, decl, moment, n_dip):
        self.__n_dip = n_dip
        group = np.zeros((n_dip+1, 3))
        
        for i in range(n_dip):
            Individuo_Dipolar.__init__(self, x, y, z, incl, decl, moment)


dipole = Individuo_Dipolar(1, 1, 1, 1, 1, 10)

dipole.dipole_group(1, 1, 1, 1, 1, 10, 2)

print(dipole.__dict__)
"""



class Dipole:

    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z

    @classmethod
    def gera_dipole(cls, x, y, z):
        pop = np.zeros((1, 3))
        pop[0,0] = x
        pop[0,1] = y
        pop[0,2] = z

        return pop


dipol = Dipole.gera_dipole(1, 1, 1)

print(dipol)



