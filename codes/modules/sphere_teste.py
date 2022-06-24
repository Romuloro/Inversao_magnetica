# --------------------------------------------------------------------------------------------------
# Title: Grav-Mag Codes
# Author: Nelson Ribeiro Filho
# Description: Source codes that will be necessary during the masters course.
# Collaborator: Rodrigo Bijani
# --------------------------------------------------------------------------------------------------

# Import Python libraries
from numba import jit
import numpy as np
# Import my libraries
import auxiliars as aux


@jit(nopython=True)
def sphere_bx(x, y, z, sphere, m, incs, decs):

    '''    
    It is a Python implementation for a Fortran subroutine contained in Blakely (1995). 
    It computes the X component of the magnetic induction caused by a sphere with uniform  
    distribution of magnetization. The direction X and Y represents the north and east, Z 
    represents growth downward. This function receives the coordinates of the points of 
    observation (X, Y, Z - arrays), the coordinates of the center of the sphere (Xe, Ye, Ze), 
    the magnetization intensity M and the values for inclination and declination (in degrees). 
    The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    sphere[0, 1, 2] - arrays - position of the center of the sphere
    sphere[3] - float - value for the spehre radius  
    sphere[4] - flaot - magnetization intensity value
    direction - numpy array - inclination and declination values
    
    Outputs:
    Bx - induced field on X direction
     
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can 
    be a set of points.    
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Calculates some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    #Setting some constants
    xe, ye, ze = sphere[0], sphere[1], sphere[2]
    #radius = sphere[3]
    
    # Distances in all axis directions - x, y e z
    rx = x - xe
    ry = y - ye
    rz = z - ze
    
    # Computes the distance (r) as the module of the other three components
    r2 = rx**2 + ry**2 + rz**2
        
    # Computes the magnetization values for all directions
    mx, my, mz = aux.dircos(incs, decs)
    
    # Auxiliar calculation
    dot = rx*mx + ry*my + rz*mz  # Scalar product
    #m = (4.*np.pi*(radius**3)*mag)/3.    # Magnetic moment
    
    # Component calculation - Bx
    bx = m*(3.*dot*rx - (r2*mx))/(r2**(2.5))

    # Final component calculation
    bx *= cm*t2nt
    
    # Return the final output
    return bx


@jit(nopython=True)
def sphere_by(x, y, z, sphere, m, incs, decs):

    '''    
    It is a Python implementation for a Fortran subroutine contained in Blakely (1995). It 
    computes the Y component of the magnetic induction caused by a sphere with uniform  
    distribution of magnetization. The direction X represents the north and Z represents 
    growth downward. This function receives the coordinates of the points of observation 
    (X, Y, Z - arrays), the coordinates of the center  of the sphere (Xe, Ye, Ze), the 
    magnetization intensity M and the values for inclination and declination (in degrees). 
    The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    sphere[0, 1, 2] - arrays - position of the center of the sphere
    sphere[3] - float - value for the spehre radius  
    sphere[4] - flaot - magnetization intensity value
    direction - numpy array - inclination, declination values
    
    Outputs:
    By - induced field on Y direction
     
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can be a 
    set of points.    
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
        
    # Calculates some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant

    #Setting some constants
    xe, ye, ze = sphere[0], sphere[1], sphere[2]
    #radius = sphere[3]
    
    # Distances in all axis directions - x, y e z
    rx = x - xe
    ry = y - ye
    rz = z - ze
    
    # Computes the distance (r) as the module of the other three components
    r2 = rx**2 + ry**2 + rz**2
        
    # Computes the magnetization values for all directions
    mx, my, mz = aux.dircos(incs, decs)
    
    # Auxiliars calculations
    dot = rx*mx + ry*my + rz*mz  # Scalar product
    #m = (4.*np.pi*(radius**3)*mag)/3.    # Magnetic moment
    
    # Component calculation - By
    by = m*(3.*dot*ry - (r2*my))/(r2**(2.5))
    
    # Final component calculation
    by *= cm*t2nt
    
    # Return the final output
    return by


@jit(nopython=True)
def sphere_bz(x, y, z, sphere, m, incs, decs):

    '''    
    It is a Python implementation for a Fortran subroutine contained in Blakely (1995). It 
    computes the Z component of the magnetic induction caused by a sphere with uniform  
    distribution of magnetization. The direction X represents the north and Z represents 
    growth downward. This function receives the coordinates of the points of observation 
    (X, Y, Z - arrays), the coordinates of the center of the sphere (Xe, Ye, Ze), the
    magnetization intensity M and the values for inclination and declination (in degrees). 
    The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    sphere[0, 1, 2] - arrays - position of the center of the sphere
    sphere[3] - float - value for the spehre radius  
    sphere[4] - flaot - magnetization intensity value
    direction - numpy array - inclination and declination values
    
    Outputs:
    Bz - induced field on Z direction
     
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can be a 
    set of points.
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Calculates some constants
    t2nt = 1.e9 # Testa to nT - conversion
    cm = 1.e-7  # Magnetization constant
    
    #Setting some constants
    xe, ye, ze = sphere[0], sphere[1], sphere[2]
    #radius = sphere[3]
    
    # Distances in all axis directions - x, y e z
    rx = x - xe
    ry = y - ye
    rz = z - ze
    
    # Computes the distance (r) as the module of the other three components
    r2 = rx**2 + ry**2 + rz**2
    
    # Computes the magnetization values for all directions
    mx, my, mz = aux.dircos(incs, decs)
    
    # Auxiliars calculations
    dot = (rx*mx) + (ry*my) + (rz*mz)  # Scalar product
    #m = (4.*np.pi*(radius**3)*mag)/3.    # Magnetic moment
    
    # Component calculation - Bz
    bz = m*(3.*dot*rz - (r2*mz))/(r2**(2.5))

    # Final component calculation
    bz *= cm*t2nt
    
    # Return the final output
    return bz


@jit(nopython=True)
def sphere_tf(x, y, z, sphere, m, F, incf, decf, incs = None, decs = None):
    
    '''    
    This function computes the total field anomaly produced due to a solid sphere, which has 
    its center located in xe, ye and ze, radius equals to r and also the magnetic property 
    (magnetic intensity). This function receives the coordinates of the points of observation 
    (X, Y, Z - arrays), the elements of the sphere, the values for inclination, declination and 
    azimuth (in one array only!) and the elements of the field (intensity, inclination, declination 
    and azimuth - IN THAT ORDER!). The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    sphere[0, 1, 2] - arrays - position of the center of the sphere
    sphere[3] - float - value for the spehre radius  
    sphere[4] - float - magnetization intensity value
    direction - numpy array - inclination and declination values
    field - numpy array - values for the field and its orientations
    
    Outputs:
    totalfield - numpy array - calculated total field anomaly
    
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can be a set of points.    
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
       
    # Compute de regional field    
    Fx, Fy, Fz = aux.regional(F, incf, decf)
    
    if incs == None:
        incs = incf
    if decs == None:
        decs = decf
    
    # Computing the components and the regional field
    bx = sphere_bx(x, y, z, sphere, m, incs, decs) + Fx
    by = sphere_by(x, y, z, sphere, m, incs, decs) + Fy
    bz = sphere_bz(x, y, z, sphere, m, incs, decs) + Fz
    
    # Final value for the total field anomaly
    tf = np.sqrt(bx**2 + by**2 + bz**2) - F
    
    # Return the final output
    return tf


@jit(nopython=True)
def sphere_tfa(x, y, z, sphere, m, incf, decf, incs = None, decs = None):

    '''    
    This function computes the total field anomaly produced due to a solid sphere, which has 
    its center located in xe, ye and ze, radius equals to r and also the magnetic property 
    (magnetic intensity). This function receives the coordinates of the points of observation 
    (X, Y, Z - arrays), the elements of the sphere, the values for inclination, declination 
    and azimuth (in one array only!) and the elements of the field (intensity, inclination, 
    declination and azimuth - IN THAT ORDER!). The observation values are given in meters.
    
    Inputs: 
    x, y, z - numpy arrays - position of the observation points
    sphere[0, 1, 2] - arrays - position of the center of the sphere
    sphere[3] - float - value for the spehre radius  
    sphere[4] - flaot - magnetization intensity value
    direction - numpy array - inclination and declination values
    field - numpy array - inclination and declination values for the field
    
    Outputs:
    tf_aprox - numpy array - approximated total field anomaly
    
    Ps. The value for Z can be a scalar in the case of one depth, otherwise it can be a 
    set of points.    
    '''
    
    # Stablishing some conditions
    if x.shape != y.shape:
        raise ValueError("All inputs must have same shape!")
    
    # Compute de regional field    
    fx, fy, fz = aux.dircos(incf, decf)
    
    if incs == None:
        incs = incf
    if decs == None:
        decs = decf
    
    # Computing the components and the regional field
    bx = sphere_bx(x, y, z, sphere, m, incs, decs)
    by = sphere_by(x, y, z, sphere, m, incs, decs)
    bz = sphere_bz(x, y, z, sphere, m, incs, decs)
    
    # Final value for the total field anomaly
    tf_aprox = fx*bx + fy*by + fz*bz
    
    # Return the final output
    return tf_aprox

