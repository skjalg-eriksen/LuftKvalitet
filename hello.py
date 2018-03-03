from cloudant import Cloudant
from flask import Flask, render_template, request, jsonify, send_file
from urllib2 import Request, urlopen, URLError
import atexit
import cf_deployment_tracker
import os
import json
import math

import numpy as np
from pandas.io.json import json_normalize
from pylab import *
import numpy as np
from pandas import DataFrame, Series
from scipy.spatial.distance import pdist, squareform
import pyproj
import utm

import dataset
from semivariogram import SVh, SV, C
from spherical_mod import spherical, opt, cvmodel
from krige import krige

#map corners
LEFT = 593065.1648494017; 
BOTTOM = 6638524.509011956;
RIGHT = 605365.439142052; TOP = 6648891.652304975;


# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

@app.route('/')
def home():
    return render_template('index.html')
    
@app.route('/data')
def data():
  # fetches data from nilu
  data = urlopen('https://api.nilu.no/aq/utd.json?areas=Oslo').read()
  # turns it into a json object
  jdata = json.loads(data)
  # puts it in a dataframe
  fdata = json_normalize(jdata)
  # picks out the wanted values
  # fdata = DataFrame(fdata, columns=['latitude','longitude', 'value', 'unit', 'component'] )
  return fdata.to_html()

@app.route('/data_min')
def data_min():
  # fetches data from nilu
  data = urlopen('https://api.nilu.no/aq/utd.json?areas=Oslo').read()
  # turns it into a json object
  jdata = json.loads(data)
  # puts it in a dataframe
  fdata = json_normalize(jdata)
  # picks out the wanted values
  fdata = DataFrame(fdata, columns=['latitude','longitude', 'value', 'unit', 'component'] )
  return fdata.to_html()

@app.route('/data_min_xy')
def data_min_xy():
   fdata = dataset.data()
   return fdata.to_html()

@app.route('/plot_data')
def plot_data():
  clf();
  fdata = dataset.data()
  #fdata = np.array( fdata[['x','y','value']] )
  fig, ax = subplots()
  ax.scatter(fdata.x, fdata.y, c=fdata.value, cmap='gray')
  #ax.scatter(fdata[0], fdata[1], c=fdata[2], cmap='gray');
  ax.set_aspect(1)
  xlim( 0, RIGHT-LEFT)
  ylim( 0, TOP-BOTTOM)
  xlabel('Easting, x [m]')
  ylabel('Northing, y [m]')
  title('value air polution') ;
  savefig('test.png',fmt='png',dpi=200)
  print(fdata)
  return send_file('test.png', mimetype='image/gif')
 
@app.route('/semiovariogram_plot')
def semiovariogram_plot():
  clf();
  z = dataset.data()
  P = np.array( z[['x','y','value']] )
  # bandwidth, plus or minus 250 meters
  bw = 500
  # lags in 500 meter increments from zero to 10,000
  hs = np.arange(0,13050,bw)
  sv = SV( P, hs, bw )
  fig, ax = subplots()
  plot(sv[0], sv[1], '.-')
  xlabel('lag [m]')
  ylabel('semivariance')
  title('Sample Semivariance')
  savefig('test_semivariogram.png',fmt='png',dpi=200)
  
  return send_file('test_semivariogram.png', mimetype='image/gif')
  
@app.route('/spherical_model_plot')
def spherical_model_plot():
  z = dataset.data()
  P = np.array( z[['x','y','value']] )
  # bandwidth, plus or minus 250 meters
  bw = 500
  # lags in 500 meter increments from zero to 10,000
  hs = np.arange(0,13050,bw)
  sv = SV( P, hs, bw )
  sp = cvmodel( P, model=spherical, hs=np.arange(0,13050, 500), bw=500 )
  fig, ax = subplots()
  clf();
  plot( sv[0], sv[1], '.-' )
  plot ( sv[0], sp( sv[0] ) ) ;
  title ('Spherical Model')
  ylabel('Semivariance')
  xlabel('Lag [m]')
  savefig('test_semivariogram_model.png', fmt='png',dpi=200)

  return send_file('test_semivariogram_model.png', mimetype='image/gif')
  
@app.route('/kriging_plot')
def kriging_plot():
  z = dataset.data()
    
  P = np.array( z[['x','y','value']] )
  # bandwidth, plus or minus 250 meters
  bw = 500
  # lags in 500 meter increments from zero to 10,000
  #hs = np.arange(0,13050,bw)
  hs = np.arange(0,RIGHT,bw)
  sv = SV( P, hs, bw )
  #sp = cvmodel( P, model=spherical, hs=np.arange(0,13050, 500), bw=500 )
  
  # calculate the kriging estimate at a number of unsampled points
  X0, X1 = P[:,0].min(), P[:,0].max()
  Y0, Y1 = P[:,1].min(), P[:,1].max()
  
  Z = np.zeros((80,100))
  dx, dy = (X1-X0)/100.0, (Y1-Y0)/80.0
  
  for i in range(80):
      print (i),
      for j in range(100):
         Z[i,j] = krige( P, spherical, hs, bw, (dy*j,dx*i), 8 )

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
  ax.scatter( z.x/200.0, z.y/200.0, facecolor='none', linewidths=0.75, s=50 )
  xlim(0, 99) ; ylim(0, 80)
  xticks( [25,50,75], [5000,10000,15000] )
  yticks( [25,50,75], [5000,10000,15000] )
   
  savefig( 'krigingpurple.png', fmt='png', dpi=200 )


  return 'work in progress, kriging'
  

@app.route('/hello')
def hello():
  return 'hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
