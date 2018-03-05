from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

def SVh( P, h, bw ):
    '''
    Experimental semivariogram for a single lag
    '''
    pd = squareform( pdist( P[:,:2] ) )
    N = pd.shape[0]
    Z = list()
    for i in range(N):
        for j in range(i+1,N):
            if( pd[i,j] >= h-bw )and( pd[i,j] <= h+bw ):
                Z.append( ( P[i,2] - P[j,2] )**2.0 )
    return np.sum( Z ) / ( 2.0 * len( Z ) )

def SV (P, hs, bw ):
    
    # Experimental variogram for a collection of lags

    sv = list()
    for h in hs:
        sv.append( SVh(P, h, bw) )
    sv = [ [ hs[i], sv[i] ] for i in range( len( hs ) ) if sv[i] > 0 ]

    return np.array( sv ).T

def C( P, h, bw ):

    # Calculate the still, covariance function

    c0 = np.var( P[:,2] )
    if h == 0:
        return c0
    return c0-SVh( P, h, bw)
