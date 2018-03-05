from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

import dataset

# fetching/parseing data
#z = open ( 'ZoneA.dat', 'r' ).readlines()
#z = [ i.strip().split() for i in z [10:] ]
#z = np.array( z, dtype=np.float )
#z = DataFrame( z, columns=['x','y','thk','por','prem','lprem','lpermp', 'lpermr'] )
z = dataset.data();
print(z)
'''
# graphing data
fig, ax = subplots()
ax.scatter(z.x, z.y, c=z.por, cmap='gray')
ax.set_aspect(1)
xlim(-1500, 22000)
ylim(-1500, 17500)
xlabel('Easting [m]')
ylabel('Northing [m]')
title('Porosity %') ;
savefig('sample_porosity.png',fmt='png',dpi=200)
'''

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

P = np.array( z[['x','y','value']] )
# part of our data set recording porosity
P = np.array( z[['x','y','value']] )
# bandwidth, plus or minus 250 meters
bw = 500
# lags in 500 meter increments from zero to 10,000
# hs = np.arange(0,10500,bw)
hs = np.arange(0,605365.439142052, bw)
sv = SV( P, hs, bw )

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
    print('\n')
    print(K)
    print('\n')
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

# calculate the kriging estimate at a number of unsampled points
X0, X1 = P[:,0].min(), P[:,0].max()
Y0, Y1 = P[:,1].min(), P[:,1].max()
nx = 10
ny = 15
Z = np.zeros((ny,nx))
dx, dy = (X1-X0)/float(nx), (Y1-Y0)/float(ny)
for i in range(nx):
    print (i),
    for j in range(ny):
       Z[j,i] = krige( P, spherical, hs, bw, (dy*j,dx*i), 5 )

cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.5, 225/255., 225/255. ),
                   (0.75, 0.141, 0.141 ),
                   (1.0, 0.0, 0.0)),
         'green': ((0.0, 1.0, 1.0),
                   (0.5, 57/255., 57/255. ),
                   (0.75, 0.0, 0.0 ),
                   (1.0, 0.0, 0.0)),
         'blue':  ((0.0, 0.376, 0.376),
                   (0.5, 198/255., 198/255. ),
                   (0.75, 1.0, 1.0 ),
                   (1.0, 0.0, 0.0)) }

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)

fig, ax = subplots()
H = np.zeros_like( Z )
for i in range( Z.shape[0] ):
    for j in range( Z.shape[1] ):
        H[i,j] = np.round( Z[i,j]*3 )

ax.matshow( H, cmap=my_cmap, interpolation='nearest' )
ax.scatter( z.x, z.y, facecolor='none', linewidths=0.75, s=50 )
#xlim(0,nx) ; ylim(0,ny)
#xticks( [25,50,75], [5000,10000,15000] )
#yticks( [25,50,75], [5000,10000,15000] )

fig.show() 
savefig( 'krigingpurple.png', fmt='png', dpi=200 )

