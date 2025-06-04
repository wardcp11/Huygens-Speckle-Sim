from numpy import *
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import logging

pool_size = mp.cpu_count()

startTime = time.time()

#*************************
##PARAMETERS##


#Wavelength
lam = 1

#Sampling Points: 2 * NMAX + 1
NMAX = 20

#Surface Sampling
dx = lam / 5
dy = dx

#Beam Waist
W0= 2*lam

#Observation Plane Sampling
dx_obs = lam / 5
dy_obs = dx_obs

#Observation Distance
dd = 50 * lam

#Surface Roughness
hmax = 5 * lam

##INPUT BEAM SHAPE
#Possible Shapes:
#   gaussian
beamShape = 'gaussian'


##END OF PARAMETERS##
#*************************


k0 = 2 * pi / lam


#Surface Sampling Points
X = linspace(-NMAX * dx, NMAX * dx, 2*NMAX + 1)
Y = linspace(-NMAX * dy, NMAX * dy, 2*NMAX + 1)

#Observation Plane Sampling Points
X_obs = linspace(-NMAX * dx_obs, NMAX * dx_obs, 2*NMAX + 1)
Y_obs = linspace(-NMAX * dy_obs, NMAX * dy_obs, 2*NMAX + 1)

#Number of Points on Surface Sampling
horizSize = X.shape[0]
verticSize = Y.shape[0]

#Number of Points in Observation Plane
horizSize_obs = X_obs.shape[0]
verticSize_obs = Y_obs.shape[0]

#Tuple for Surface Sizing
viewingWindow = (horizSize, verticSize)

#Tuple for Observation Sizing
viewingWindow_obs = (horizSize_obs, verticSize_obs)

#Filler To Ensure Correct Type within 4d Structure
filler = zeros(viewingWindow_obs, dtype=complex128)

def getTime():
    """Prints elapsed time since global `startTime` in seconds.
    
    Formats output to 2 decimal places.
    """

    print('Time [s]: ' + '{0:.2f}'.format(time.time() - startTime))

def D(X,Y, x,y,z):
    """Calculates 3D Euclidean distance between two points.
    
    Args:
        X (float): Source point x-coordinate
        Y (float): Source point y-coordinate
        x (float): Target point x-coordinate
        y (float): Target point y-coordinate
        z (float): Z-axis separation between points
        
    Returns:
        float: Distance between (X,Y,0) and (x,y,z)
    """
    
    return sqrt((x-X)**2 + (y-Y)**2 + z**2)

def A(X,Y, w0):
    """Input amplitude distribution for beam source.
    
    Supports 'gaussian' beam profile (configured via global `beamShape`).
    
    Args:
        X (float): X-coordinate in source plane
        Y (float): Y-coordinate in source plane
        w0 (float): Beam waist radius (for Gaussian beam)
        
    Returns:
        float: Amplitude value at (X, Y)
    """

    if beamShape == 'gaussian':
        numerator = -1 * (X**2 + Y**2)
        denominator = 2 * w0**2
        return exp(numerator / denominator)

def pointEE(X,Y, obsX,obsY):
    """Calculates single-point electric field contribution using Huygens' principle.
    
    Combines amplitude distribution and spherical wave propagation.
    
    Args:
        X (float): Source point x-coordinate
        Y (float): Source point y-coordinate
        obsX (float): Observer point x-coordinate
        obsY (float): Observer point y-coordinate
        
    Returns:
        complex: Complex electric field contribution at (obsX, obsY)
    """

    numerator = exp(1j * k0 * D(X,Y, obsX,obsY,dd))
    denominator = D(X,Y, obsX,obsY,dd)
    return A(X,Y, W0) * numerator / denominator

def assignEE(tup):
    """Computes E-field contributions for a source point.
    
    Maps source coordinates to observer grid. Designed for multiprocessing.
    
    Args:
        tup (tuple): (m, n) indices for source grid coordinates
    
    Returns:
        tuple: 
            - EfromXY (ndarray): Complex E-field matrix from source (X[m], Y[n])
            - tup (tuple): Input indices (for result tracking)
    """

    EfromXY = zeros(viewingWindow, dtype=complex128)
    for m1 in range(horizSize_obs):
        for n1 in range(verticSize_obs):
            EfromXY[m1][n1] = pointEE(X[tup[0]],Y[tup[1]], X_obs[m1],Y_obs[n1])

    return EfromXY, tup

def computeEfield():
    """Computes total electric field using parallelized Huygens' principle.
    
    Distributes source point calculations across multiprocessing pool.
    Populates a 4D object array of complex field matrices.
    
    Returns:
        ndarray: 4D array of complex E-field matrices from all sources
    """
    Efield = ndarray(viewingWindow, object) #Electric Field
    for m in range(horizSize):
        for n in range(verticSize):
            Efield[m][n] = filler

    combinations = [(m, n) for m in range(horizSize) for n in range(verticSize)]

    with mp.Pool(processes=pool_size) as p:
        it = p.imap(assignEE, combinations)
        for result, resTup in it:
            print('Completed:',resTup)
            Efield[resTup[0]][resTup[1]] = result

    return Efield

def computeAmplitude():
    """Calculates source amplitude distribution over the beam grid.
    
    Returns:
        ndarray: 2D amplitude matrix over (horizSize, verticSize) grid
    """
    AA = zeros(viewingWindow) #Amplitude Distribution
    for m in range(horizSize):
        for n in range(verticSize):
            AA[m][n] = A(X[m], Y[n], W0)

def sumEField(Efield):
    """Synthesizes total E-field by summing contributions with phase shifts.
    
    Includes z-dependent phase term (global `ZZ` matrix) in summation.
    
    Args:
        Efield (ndarray): 2D array of complex E-field matrices
        
    Returns:
        ndarray: Complex total E-field matrix over observation grid
    """
    ETOT = zeros(viewingWindow, dtype=complex128) #TOTAL Electric Field
    for m in range(horizSize):
        print('Creating ETOT:',m)
        for n in range(verticSize):
            ETOT += Efield[m][n] * exp(1j * 2 * k0 * ZZ[m][n])
    return ETOT

if __name__ == "__main__":

    AA = computeAmplitude()

    ZZ = random.rand(horizSize,verticSize) * hmax #RANDOM SURFACE

    Efield = computeEfield()

    ETOT = sumEField(Efield)

    getTime()

    plt.imshow(abs(ETOT))
    plt.show()