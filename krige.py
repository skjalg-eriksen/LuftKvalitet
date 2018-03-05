from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

from semivariogram import SVh, SV, C
from spherical_mod import spherical, opt, cvmodel

def krige( P, model, hs, bw, u, N ):
    '''
    Input       (P)     ndarray, data
                (model) modeling function
                         - spherical
                         - exponential
                         - gaussian
                (hs)    kriging distances
                (bw)    krigin bandwidth
                (u)     unsampled point
                (N)     number of neighboring
                        points to consider
    '''

    # covariance function
    covfct = cvmodel( P, model, hs, bw )
    # mean of the variable
    mu = np.mean( P[:,2] )

    # distance between u and each data point in P
    d = np.sqrt( ( P[:,0] - u[0] )**2.0 + ( P[:,1] - u[1] )**2.0 )
    # add these distances to P
    P = np.vstack(( P.T, d )).T
    # sort P by these distances
    # take the first N of them
    P = P[d.argsort()[:N]]

    # apply the covariance model to the distances
    k = covfct( P[:,3] )
    # cast as a matrix
    k = np.matrix( k ).T
 
    # form a matrix of distances between existing data points
    K = squareform( pdist( P[:,:2] ) )
    # apply the covariance model to these distances
    K = covfct( K.ravel() )
    # re-cast as a NumPy array -- thanks M.L.
    K = np.array( K )
    # reshape into an array
    K = K.reshape(N,N)
    # cast as a matrix
    K = np.matrix( K )
 

    # calculate the krigin weights
    weights = np.linalg.inv( K ) * k
    weights = np.array( weights )

    # calculate the residuals
    residuals = P[:,2] - mu

    # calculate the estimation
    estimation = np.dot( weights.T, residuals ) + mu

    return float ( estimation )
