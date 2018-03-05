from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

from semivariogram import SVh, SV, C

def opt ( fct, x, y, C0, parameterRange=None, meshSize=1000 ):

    #determining the optimal value a for the spherical model.
    
    if parameterRange == None:
        parameterRange =  [x[1], x[-1] ]
        
    mse = np.zeros( meshSize )
    a = np.linspace( parameterRange[0], parameterRange[1], meshSize )
    for i in range( meshSize ):
        mse[i] = np.mean( ( y - fct( x, a[i], C0 ) )**2.0 )
    return a[ mse.argmin() ]

def spherical ( h, a, C0 ):
    # Spherical model of the semivariogram

    # if h is a single digit
    if type(h) == np.float64:
        # calculate the spherical function
        if h <= a:
            return C0*( 1.5*h/a - 0.5*(h/a)**3.0 )
        else:
            return C0
    # if h is an iterable
    else:
        # calculate the spherical function for all elements
        a = np.ones( h.size ) * a
        C0 = np.ones( h.size ) * C0
        return list(map( spherical, h, a, C0 ))

def cvmodel ( P, model, hs, bw ):
    '''
    Input:  (P)         ndarray, data
            (model)     modeling function
                         - spherical
                         - exponential
                         - gaussian
            (hs)        distances
            (bw)        bandwidth
    Output: (covfct)    function modeling the covariance
    '''
    # calculate the semivariogram
    sv = SV(P, hs, bw )
    # calculate the sill
    C0 = C( P, hs[0], bw )
    # calculate the optimal parameters
    param = opt ( model, sv[0], sv[1], C0 )
    # return a covariance function
    covfct = lambda h, a=param: model( h, a, C0 )
   # covfct = lambda h, a=param: model( h, a, C0 ) # this makes a unflippd curve compared to the tutorial, found in comments
    return covfct

