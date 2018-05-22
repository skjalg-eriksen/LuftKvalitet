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
from krige import SVh, SV, C, spherical, opt, cvmodel, krige

import numpy as np
from pylab import *
from pandas import DataFrame, Series
import utm;


# gps coordinates for map
# TOP LEFT      59.963886, 10.662291
# TOP RIGHT     59.963886, 10.887139
# BOTTOM RIGHT  59.873800, 10.887139
# BOTTOM LEFT   59.873800, 10.662291

# map corners in utm
LEFT    = 593065.1648494017;  # X0 for the map in utm
BOTTOM  = 6638524.509011956;  # Y0 for the map in utm
RIGHT   = 605365.439142052;   # X1 for the map in utm
TOP     = 6648891.652304975;  # Y1 for the map in utm

def krige_task(data):
  # dataframe containg all the messurements from dataset
  z = data
  
  #costume points to improve our analysis
  num_points = len(z.index)
  unit_type = z.loc[1, 'unit']
  comp_type = z.loc[1, 'component']

  lowest_point = 900000  # just some large number so that we can find the lowest in the table
  #lowest value
  for i, row in z.iterrows():
    if lowest_point > z.loc[i, 'value']:
      lowest_point = z.loc[i, 'value']
      
  # make the costume points to be 40% of the lowest point
  new_low = lowest_point*0.4

  # bottom left corner in GPS latitude and longitude
  lat1 = 59.873800
  long1 = 10.662291
  
  # all gps points of costume points
  # 59.878971, 10.678320
  # 59.890065, 10.717549
  # 59.880442, 10.746924
  # 59.874472, 10.871463
  # 59.957640, 10.820121
  # 59.902286, 10.871526
  
  # array of the points
  lant_arr = [59.878971, 59.890065, 59.880442, 59.874472, 59.957640, 59.902286]
  lat_arg  = [10.678320, 10.717549, 10.746924, 10.871463, 10.820121, 10.871526]
  
  t = 0; # loop index
  for i in lant_arr:
    x = utm.from_latlon( lant_arr[t], lat_arg[t] )[0]  - utm.from_latlon(lat1, long1)[0];
    y = utm.from_latlon( lant_arr[t], lat_arg[t] )[1] - utm.from_latlon(lat1, long1)[1];
    new_df = [lant_arr[t],lat_arg[t], new_low, unit_type, comp_type, x, y]
    t = t+1
    z.loc[len(z)] = new_df
  
  # part of our data set recording pollution
  P = np.array( z[['x','y','value']] )
  # bandwidth
  bw = 15500
  # kriging distances
  hs = np.arange(0,20000, bw)

  X0, X1 = 0, RIGHT-LEFT
  Y0, Y1 = 0, TOP-BOTTOM

  # resolution of the interpolated picture, indexes for the numpey array
  nx = 48
  ny = 60
  
  # consider all the points available, but no more than 10.
  num_points = len(z.index)
  if (num_points > 10):
    num_points = 10;
  
  # make an numpy array
  Z = np.zeros((ny,nx))
  
  # divide the range of the map width, height with our resolution
  dx, dy = (X1-X0)/float(nx), (Y1-Y0)/float(ny)
  
  # calculate each point with kriging
  for i in range(nx):
      print (i), # print i for showing progress, goes till it reaches nx
      for j in range(ny):
          # get the x, y address of the cell for the numpey array
          x = X0 + i * dx
          y = Y0 + j * dy
          # preform kriging for the cell
          Z[ j, i ] = krige( P, spherical, hs, bw, (x, y), num_points )

  # round the calculations, to increase the contrast
  H = np.zeros_like( Z )
  for i in range( Z.shape[0] ):
      for j in range( Z.shape[1] ):
          H[i,j] = np.round( Z[i,j] )

  return H;