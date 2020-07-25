import numpy as np

def f_difference (dado_referencia, dado_calculado):
    std = np.std(dado_referencia)
    dif = (dado_referencia - dado_calculado)**2 / (std**2)
    rms = np.sum(dif) / len(dif)
    
    return rms


def sample_random_coordinated(dicionario):
    xmax = dicionario.get('xmax')
    xmin = dicionario.get('xmin')
    ymax = dicionario.get('ymax')
    ymin = dicionario.get('ymin')
    zlim = dicionario.get('zlim')
    n = dicionario.get('n')
    
    resultadox=[]
    resultadoy=[]
    resultadoz=[]
#---------------------------------------------------------------------------------------------------------------------#
    for i in range(n):
        sorted_x1, sorted_y1,sorted_z1 = (np.random.uniform(xmin, xmax),
                                      np.random.uniform(ymin, ymax),
                                      np.random.uniform(0.0, zlim))
        resultadox.append(sorted_x1)
        resultadoy.append(sorted_y1)
        resultadoz.append(sorted_z1)  
#---------------------------------------------------------------------------------------------------------------------#        
    return resultadox, resultadoy, resultadoz

def sample_random_mag(dicionario, homogeneo):
    inclmax = mag_bounds.get('inclmax')
    inclmin = mag_bounds.get('inclmin')
    declmax = mag_bounds.get('declmax')
    declmin = mag_bounds.get('declmin')
    magmax = mag_bounds.get('mag_max')
    magmin = mag_bounds.get('mag_min')
    n = mag_bounds.get('n')
#---------------------------------------------------------------------------------------------------------------------#
    incl=[]
    decl=[]
    mag=[]
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
            sorted_incl, sorted_decl, sorted_mag = (np.random.uniform(inclmax,inclmin),
                                                    np.random.uniform(declmax, declmin),
                                                    np.random.uniform(magmax, magmin))
            incl.append(sorted_incl)
            decl.append(sorted_decl)
            mag.append(sorted_mag)
    
    return incl, decl, mag

def tfa_n_dots(dicionario1, dicionario2):
    #Dicionário com os valores de mag
    incl = dicionario1.get('incl')
    decl = dicionario1.get('decl')
    mag = dicionario1.get('mag')
    n = dicionario1.get('n')
#---------------------------------------------------------------------------------------------------------------------#
    #Dicionário com os valores de coordenadas
    Xref = dicionario2.get('Xref')
    Yref = dicionario2.get('Yref')
    Zref = dicionario2.get('Zref')
    I = dicionario2.get('I')
    D = dicionario2.get('D')
    coodX = dicionario2.get('coodX')
    coodY = dicionario2.get('coodY')
    coodZ = dicionario2.get('coodZ')
    raio = dicionario2.get('raio')
#---------------------------------------------------------------------------------------------------------------------#  
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