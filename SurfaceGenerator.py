import matplotlib.pyplot as plt
from numpy import *
import scipy.integrate as integrate
import cupy as cp
import cupyx.scipy.signal as cps

def symmetricFFTConvolve(A, kernel,  amountofPadding):

    if amountofPadding == 0:
        return cps.fftconvolve(A, kernel, mode='same')
    else:
        for n in range(amountofPadding):
            if n == 0:
                Middle = A
            else:
                Middle = padded

            FlippedHoriz = Middle[:, ::-1]
            FlippedVert = Middle[::-1, :]
            FlippedBOTH = FlippedHoriz[::-1, :]

            temp1 = cp.concatenate((FlippedBOTH, FlippedVert, FlippedBOTH), axis=1)
            temp2 = cp.concatenate((FlippedHoriz, Middle, FlippedHoriz), axis=1)
            temp3 = cp.concatenate((FlippedBOTH, FlippedVert, FlippedBOTH), axis=1)

            padded = cp.concatenate((temp1, temp2, temp3), axis=0)

        lowerY = (n+1) * padded.shape[0]/(3**(n+1)) - 1
        upperY = (n+2) * padded.shape[0]/(3**(n+1)) - 1
        lowerX = (n+1) * padded.shape[1]/(3**(n+1)) - 1
        upperX = (n+2) * padded.shape[1]/(3**(n+1)) - 1

        padded = cps.fftconvolve(padded, kernel, mode='same')[lowerY:upperY, lowerX:upperX]

        return padded

def wrappedFFTConvolve(A, kernel, amountofPadding):

    if amountofPadding == 0:
        return cps.fftconvolve(A, kernel, mode='same')
    else:
        for n in range(amountofPadding):
            if n == 0:
                Middle = A
            else:
                Middle = padded

            temp1 = cp.concatenate((Middle, Middle, Middle), axis=1)
            temp2 = cp.concatenate((Middle, Middle, Middle), axis=1)
            temp3 = cp.concatenate((Middle, Middle, Middle), axis=1)
            padded = cp.concatenate((temp1, temp2, temp3), axis=0)

        lowerY = (n+1) * padded.shape[0]/(3**(n+1)) - 1
        upperY = (n+2) * padded.shape[0]/(3**(n+1)) - 1
        lowerX = (n+1) * padded.shape[1]/(3**(n+1)) - 1
        upperX = (n+2) * padded.shape[1]/(3**(n+1)) - 1

        padded = cps.fftconvolve(padded, kernel, mode='same')[lowerY:upperY, lowerX:upperX]

        return padded


    """
    Sets elements outside the given radius from the center to zero.
    Returns the modified array and the maximum value in the modified array.
    """
    rows, cols = arr.shape
    # Create coordinate grids
    y, x = cp.ogrid[:rows, :cols]
    
    # Calculate center coordinates (geometric center)
    center_y = (rows - 1) / 2.0
    center_x = (cols - 1) / 2.0
    
    # Compute Euclidean distance from center
    dist = cp.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Create mask for elements within the radius
    mask = dist <= radius
    
    # Apply mask: set elements outside radius to 0
    result = cp.where(mask, arr, 0)
    
    return result

def surfaceOUKernel(L, dx, correlationLength, padding=0):

    correlLength = correlationLength
    N = int(L/dx)

    X = cp.linspace(-L/2, L/2, N)
    Y = cp.linspace(-L/2, L/2, N)
    X,Y = cp.meshgrid(X,Y)

    kernelX = cp.linspace(-L, L, 2*N+1)
    kernelY = cp.linspace(-L, L, 2*N+1)
    kernelX,kernelY = cp.meshgrid(kernelX,kernelY)

    D = lambda x,y, X,Y: cp.sqrt((x-X)**2 + (y-Y)**2)
    OUkernel = lambda l: cp.exp(- D(0,0, kernelX,kernelY) / l)

    #kernel generation
    kernel = OUkernel(correlLength)

    #normalize kernel to maintain power
    integral_x = integrate.simpson(cp.asnumpy(kernel), dx=dx, axis=1)
    result = integrate.simpson(integral_x, dx=dx)
    kernel = kernel / result


    data = cp.random.randn(len(X), len(Y))
    surface = cp.asnumpy(wrappedFFTConvolve(data, kernel, amountofPadding=padding))

    return surface