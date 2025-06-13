from numpy import *
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import logging
import os

pool_size = mp.cpu_count()
startTime = time.time()
logger = logging.getLogger(__name__)

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
dd = 100 * lam

#Surface Roughness
hmax = .2 * lam

##INPUT BEAM SHAPE
#Possible Shapes:
#   gaussian
beamShape = 'gaussian'


##END OF PARAMETERS##
#*************************


k0 = 2 * pi / lam

dataDictionary = {}

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

def pointEE(X,Y, obsX,obsY, tup):
    
    xIndex = tup[0]
    yIndex = tup[1]
    AmplitudeArray = tup[2]
    xPoint = X[xIndex]
    yPoint = Y[yIndex]

    numerator = exp(1j * k0 * D(xPoint,yPoint, obsX,obsY,dd))
    denominator = D(xPoint,yPoint, obsX,obsY,dd)
    return AmplitudeArray[xIndex][yIndex] * numerator / denominator

def assignEE(tup):

    EfromXY = zeros(viewingWindow, dtype=complex128)
    for m1 in range(horizSize_obs):
        for n1 in range(verticSize_obs):
            EfromXY[m1][n1] = pointEE(X,Y, X_obs[m1],Y_obs[n1], tup)

    return EfromXY, (tup[0],tup[1])

def computeEfield(AmplitudeArray):

    Efield = ndarray(viewingWindow, object) #Electric Field
    for m in range(horizSize):
        for n in range(verticSize):
            Efield[m][n] = filler

    combinations = [(m, n, AmplitudeArray) for m in range(horizSize) for n in range(verticSize)]

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
    return AA

def sumEField(Efield):

    ETOT = zeros(viewingWindow, dtype=complex128) #TOTAL Electric Field
    for m in range(horizSize):
        print('Creating ETOT:',m)
        for n in range(verticSize):
            ETOT += Efield[m][n] * exp(1j * 2 * k0 * ZZ[m][n])
    return ETOT

def logData():

    logsDirExists = False
    logFile = './logs'
    for file in os.scandir('./'):
        if file.name == 'logs':
            logsDirExists = True
            break
    if logsDirExists == False:
        os.mkdir(logFile)

    newestLog = 1
    for file in os.scandir(logFile):
        if file.name == 'run_' + str(newestLog) + '.log':
            newestLog += 1

    logging.basicConfig(filename='./logs/run_' + str(newestLog) + '.log', level=logging.INFO)
    logger.info('Started')
    logger.info('--User Parameters--')
    logger.info('Distance: ' + str(dd))
    logger.info('Beam Waist: ' + str(W0))
    logger.info('Roughness: ' + str(hmax))
    logger.info('Surface Sample Size: ' + str(dx))
    logger.info('Observation Sample Size: ' + str(dx_obs))
    logger.info("NMAX: " + str(NMAX))
    logger.info('--End User Parameters--')
    logger.info("Runtime: " + str(getTime()))

def saveData(dataDict):

    logData()
    dataFolder = './logs'

    newestData = 1
    for file in os.scandir(dataFolder):
        if file.name == str(newestData):
            newestData += 1
    os.mkdir(dataFolder + '/' + str(newestData))

    for key in dataDict:
        for toSave in dataDict[key]:
            save(dataFolder + '/' + str(newestData) + '/' + str(key), toSave)


if __name__ == "__main__":

    AA = computeAmplitude()

    ZZ = random.rand(horizSize,verticSize) * hmax #RANDOM SURFACE
    Efield = computeEfield(AA)
    ETOT = sumEField(Efield)

    dataDictionary["Initial ETOT"] = [ETOT]
    dataDictionary["Initial AA"] = [AA]
    dataDictionary["Initial ZZ"] = [ZZ]

    if False: #Multibounce
        for bounces in range(1,1):
            AA = abs(ETOT)
            ZZ = random.rand(horizSize,verticSize) * hmax #RANDOM SURFACE
            Efield = computeEfield(abs(ETOT))
            ETOT = sumEField(Efield)

            dataDictionary["bounceETOT{0}".format(bounces)] = [ETOT]
            dataDictionary["bounceAA{0}".format(bounces)] = [AA]
            dataDictionary["bounceZZ{0}".format(bounces)] = [ZZ]
    

    if True: #DATA LOGGING
        saveData(dataDictionary)
