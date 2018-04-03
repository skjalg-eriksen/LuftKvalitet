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
import datetime

import numpy as np
from pandas.io.json import json_normalize
from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform

#map corners
LEFT = 593065.1648494017; 
BOTTOM = 6638524.509011956;
RIGHT = 605365.439142052; 
TOP = 6648891.652304975;


def krige_task(data, date): 
  z = data
  # part of our data set recording porosity
  P = np.array( z[['x','y','value']] )
  # bandwidth, plus or minus 250 meters
  bw = 15500
  # lags in 500 meter increments from zero to 10,000
  # hs = np.arange(0,10500,bw)
  hs = np.arange(0,20000, bw)
  sv = SV( P, hs, bw )

  X0, X1 = 0, RIGHT-LEFT
  Y0, Y1 = 0, TOP-BOTTOM

  nx = 48
  ny = 60
  num_points = len(z.index)
  Z = np.zeros((ny,nx))
  dx, dy = (X1-X0)/float(nx), (Y1-Y0)/float(ny)
  for i in range(nx):
      print (i),
      for j in range(ny):
          x = X0 + i*dx
          y = Y0 + j*dy
          Z[j,i] = krige( P, spherical, hs, bw, (x, y), num_points )

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

  
  
  H = np.zeros_like( Z )
  for i in range( Z.shape[0] ):
      for j in range( Z.shape[1] ):
          H[i,j] = np.round( Z[i,j]*3 )


  #image with info
  fig, ax = subplots()    
  fig.dpi=400
  ax.imshow(H, cmap=my_cmap, origin='lower', interpolation='nearest', alpha=0.7, extent=[X0, X1, Y0, Y1])

  sc = ax.scatter( z.x, z.y, cmap=my_cmap, c=z.value, linewidths=0.75, s=50 )
  plt.colorbar(sc)

  fig.suptitle('component: ' + str(z['component'].iloc[0]) + ', date: ' + str(date), fontsize=14)
  fig.savefig("info",dpi = 400)
  


  #borderless/infoless image
  fig, ax = subplots()
  ax.imshow(H, cmap=my_cmap, origin='lower', interpolation='gaussian', alpha=0.7, extent=[X0, X1, Y0, Y1])
  ax.axis('off')
  fig.dpi=400

  fig.set_size_inches(5.95, 5)
  imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig("img",dpi = 400 , bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  
  
  '''
  print("Running kriging");
  z = dataset.data();
  
  
  # part of our data set recording porosity
  P = np.array( z[['x','y','value']] )
  # bandwidth, plus or minus 250 meters
  bw = 15500
  # lags in 500 meter increments from zero to 10,000
  # hs = np.arange(0,10500,bw)
  hs = np.arange(0,20000, bw)
  sv = SV( P, hs, bw )


  X0, X1 = 0, RIGHT-LEFT
  Y0, Y1 = 0, TOP-BOTTOM


  nx = 48
  ny = 60
  num_points = 7
  Z = np.zeros((ny,nx))
  dx, dy = (X1-X0)/float(nx), (Y1-Y0)/float(ny)
  for i in range(nx):
      print (i),
      for j in range(ny):
          x = X0 + i*dx
          y = Y0 + j*dy
          Z[j,i] = krige( P, spherical, hs, bw, (x, y), num_points )

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
  fig.dpi=400
  H = np.zeros_like( Z )
  for i in range( Z.shape[0] ):
      for j in range( Z.shape[1] ):
          H[i,j] = np.round( Z[i,j]*3 )

  #ax.matshow( H, cmap=my_cmap, interpolation='nearest' )
  ax.imshow(H, cmap=my_cmap, origin='lower', interpolation='nearest', alpha=0.7, extent=[X0, X1, Y0, Y1])
  sc = ax.scatter( z.x, z.y, cmap=my_cmap, c=z.value, linewidths=0.75, s=50 )
  #xlim(0,nx) ; ylim(0,ny)
  plt.colorbar(sc)
  date =  str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
  name = 'component: ' + str(z['component'].iloc[0]) + ', date: ' + date
  savefig(str(name), fmt='png', dpi=200 )
  
  file_object  = open(str(name)+".png", "r");
  file_object = base64.b64encode(file_object.read())
  
  document = {
      'date': date, 
      'component': str(z['component'].iloc[0]),
      '_attachments': { str(z['component'].iloc[0])+"_img" : {'data': file_object}}
  }
  #connect to db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  #store document
  document = db.create_document(document);
  '''
