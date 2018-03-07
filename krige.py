'''
The MIT License (MIT)

Copyright (c) <2014> <Connor Johnson>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import matplotlib.pyplot as plt

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

'''
plot(sv[0], sv[1], '.-')
xlabel('lag [m]')
ylabel('semivariance')
title('Sample Semivariance')
savefig('sample_semivariogram2.png',fmt='png',dpi=200)
'''

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

'''
sp = cvmodel( P, model=spherical, hs=np.arange(0,10500, 500), bw=500 )
plot( sv[0], sv[1], '.-' )
plot ( sv[0], sp( sv[0] ) ) ;
title ('Spherical Model')
ylabel('Semivariance')
xlabel('Lag [m]')
savefig('semivariogram_model.png', fmt='png',dpi=200)
'''

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
    
    #print('\n')
    #print(K)
    #print('\n')
    
    # reshape into an array
    K = K.reshape(N,N);
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
