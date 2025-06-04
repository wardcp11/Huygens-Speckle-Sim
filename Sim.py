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
    print('Time [s]: ' + '{0:.2f}'.format(time.time() - startTime))

def D(X,Y, x,y,z):
    #Euclidian Distance
    
    return sqrt((x-X)**2 + (y-Y)**2 + z**2)

def A(X,Y, w0):
    #Input Amplitude Distribution

    if beamShape == 'gaussian':
        numerator = -1 * (X**2 + Y**2)
        denominator = 2 * w0**2
        return exp(numerator / denominator)

def pointEE(X,Y, obsX,obsY):
    #Naive Huygens' Principle

    numerator = exp(1j * k0 * D(X,Y, obsX,obsY,dd))
    denominator = D(X,Y, obsX,obsY,dd)
    return A(X,Y, W0) * numerator / denominator

def assignEE(tup):
    #Paralell Function

    temp = zeros(viewingWindow, dtype=complex128)
    for m1 in range(horizSize_obs):
        for n1 in range(verticSize_obs):
            temp[m1][n1] = pointEE(X[tup[0]],Y[tup[1]], X_obs[m1],Y_obs[n1])

    return temp, tup

def computeEfield():
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
    AA = zeros(viewingWindow) #Amplitude Distribution
    for m in range(horizSize):
        for n in range(verticSize):
            AA[m][n] = A(X[m], Y[n], W0)

def sumEField(Efield):
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