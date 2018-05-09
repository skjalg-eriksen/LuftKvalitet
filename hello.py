from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
from cloudant.document import Document
from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from urllib2 import Request, urlopen, URLError
import atexit
import cf_deployment_tracker
import os
import json
import math
import re
import simplejson
import io

import atexit
from apscheduler.scheduler import Scheduler

import matplotlib
matplotlib.use('Agg')

import numpy as np
from pandas.io.json import json_normalize, read_json
from pylab import *
import numpy as np
from pandas import DataFrame, Series, to_datetime
from scipy.spatial.distance import pdist, squareform
import pyproj
import utm

import dataset
from krige import SVh, SV, C, spherical, opt, cvmodel, krige
import datetime
import base64
from kriging_task import krige_task;

#map corners
LEFT = 593065.1648494017; 
BOTTOM = 6638524.509011956;
RIGHT = 605365.439142052; 
TOP = 6648891.652304975;
X0, X1 = 0, RIGHT-LEFT
Y0, Y1 = 0, TOP-BOTTOM

cron = Scheduler(daemon=True)
cron.start()

# Emit Bluemix deployment event
cf_deployment_tracker.track()

app = Flask(__name__)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))


db_name = 'luftkvalitet_db'
client = None
db = None

if 'VCAP_SERVICES' in os.environ:
    vcap = json.loads(os.getenv('VCAP_SERVICES'))
    print('Found VCAP_SERVICES')
    if 'cloudantNoSQLDB' in vcap:
        creds = vcap['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
elif os.path.isfile('vcap-local.json'):
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
        
        
#Green, yellow, orange, red;
cdict = {'red':   ((0.0, 0.0, 0.0),
                   (0.35, 1.0, 1.0),   
                   (0.4, 1.0, 1.0),   
                   (0.8, 1.0, 1.0),  
                   (1.0, 2.0, 0.0)), 

         'green': ((0.0,  1.0, 1.0), 
                   (0.5,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.1, 0.0, 0.0),  
                   (0.4, 0.0, 0.0),  
                   (1.0, 0.0, 0.0))  
          }
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)   

@app.route('/')
def home():
  return render_template('index.html')


@app.route('/img/<_id>')
def get_img(_id):
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  #get the document from the db
  doc = db[ _id ]
  
  #load the krige data
  H = simplejson.loads(doc['krige_data'])
  buffr = io.BytesIO()

  #min max values for the color
  maxvalue = 100;
  minvalue = 0;
  
  #plot the figure
  fig, ax = subplots()
  ax.imshow(H, cmap=my_cmap, vmin=minvalue, vmax=maxvalue, origin='lower', interpolation='gaussian', alpha=0.7, extent=[X0, X1, Y0, Y1])
  ax.axis('off')
  #fig.dpi=400
  
  #set proper image size and extent to plot
  fig.set_size_inches(5.95, 5)
  imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  #fig.savefig(buffr, dpi = 400, bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  fig.savefig(buffr, format='svg', bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  
  #go to start of file
  buffr.seek(0)
  
  return send_file(buffr, mimetype='image/svg+xml')


@app.route('/contour/<_id>')
def get_contour(_id):
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  #get the document from the db
  doc = db[ _id ]
  H = simplejson.loads(doc['krige_data'])
  buffr = io.BytesIO()
  
  maxvalue = 100;
  minvalue = 0;
  fig, ax = subplots()
  
  CS = plt.contour(H,vmin=minvalue, vmax=maxvalue,extend='max',  colors='royalblue', origin='lower', alpha=1, extent=[X0, X1, Y0, Y1])
  plt.clabel(CS, inline=10, fontsize=4 , fmt='%0.1f')
  
  #ax.imshow(H, cmap=my_cmap,vmin=minvalue, vmax=maxvalue, origin='lower', interpolation='gaussian', alpha=0.7, extent=[X0, X1, Y0, Y1])
  ax.axis('off')
  fig.dpi=400

  fig.set_size_inches(5.95, 5)
  imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(buffr, dpi = 400, bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  buffr.seek(0)
  
  return send_file(buffr, mimetype='image/png')


@app.route('/info/<_id>')
def get_info(_id):
  
  X0, X1 = 0, RIGHT-LEFT
  Y0, Y1 = 0, TOP-BOTTOM

  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  #get the document from the db
  doc = db[ _id ]
  H = simplejson.loads(doc['krige_data'])
  z = read_json(doc['data'], orient='index')
  date = doc['date']
 
  maxvalue = 100;
  minvalue = -50;
  
  buffr = io.BytesIO()
  fig, ax = subplots()    
  fig.dpi=400
  ax.imshow(H, cmap=my_cmap, vmin=minvalue, vmax=maxvalue, origin='lower', interpolation='nearest', alpha=0.7, extent=[X0, X1, Y0, Y1])

  sc = ax.scatter( z.x, z.y, cmap=my_cmap,vmin=minvalue, vmax=maxvalue, c=z.value, linewidths=0.75, s=50 )
  plt.colorbar(sc)

  fig.suptitle('component: ' + str(z['component'].iloc[0]) + ', date: ' + str(date), fontsize=14)
  fig.savefig(buffr, dpi = 400)
  
  buffr.seek(0)
  
  return send_file(buffr, mimetype='image/png')

@app.route('/show_data')
def show_data():
   #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  
  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', 'data', '_id'])
  #fdocs['date'] = to_datetime(fdocs['date'])
  fdocs = fdocs.reset_index(drop=True)
  fdocs.sort_values(['date', 'component'])
  #get the components
  components = fdocs['component'].unique().tolist();
  
  data = [None]* len(fdocs)
  for i, row in fdocs.iterrows():
    tmp = read_json(fdocs.loc[i, 'data'], orient='index')
    tmp = tmp.reset_index()
    data[i] = tmp
    fdocs.loc[i, 'data'] = i;
  
  #make a list of same size as components
  complist = [None]* len(components)
  for i in range(len(components)):
    #drop everything but relevant info
    tmp = fdocs.drop(fdocs[fdocs.component != components[i]].index)
    #sort them
    tmp = tmp.sort_values(['date'], ascending=[False])
    #re index the dataframe
    tmp = tmp.reset_index(drop=True)
    #put the dataframe into the list
    complist[i] = tmp;
  return render_template('show2.html', entries = complist, data = data);
  


@app.route('/all_entries')
def all_entries():
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  
  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', 'data', '_id'])
  fdocs['date'] = to_datetime(fdocs['date'])
  fdocs = fdocs.reset_index(drop=True)
  fdocs.sort_values(['date', 'component'])
  #get the components
  components = fdocs['component'].unique().tolist();
  
  data = [None]* len(fdocs)
  for i, row in fdocs.iterrows():
    tmp = read_json(fdocs.loc[i, 'data'], orient='index')
    tmp = tmp.reset_index()
    data[i] = tmp
    fdocs.loc[i, 'data'] = i;
  
  #make a list of same size as components
  complist = [None]* len(components)
  for i in range(len(components)):
    #drop everything but relevant info
    tmp = fdocs.drop(fdocs[fdocs.component != components[i]].index)
    #sort them
    tmp = tmp.sort_values(['date'], ascending=[False])
    #re index the dataframe
    tmp = tmp.reset_index(drop=True)
    #put the dataframe into the list
    complist[i] = tmp;
  return render_template('entries.html', entries = complist, data = data);
  
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
  #date =  str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
  data = dataset.data()
  data.sort_values(by=['component'])
  #data.set_index(keys=['component'], drop=False,inplace=True)
  names = data['component'].unique().tolist()

  for component in names:
    compdata = data.loc[data.component == component]
    compdata = compdata.reset_index(drop=True)
    if(4 >len(compdata.index)):
      continue;
    
    krige_data = krige_task( compdata[['latitude','longitude', 'value', 'unit', 'component', 'x', 'y']] )
    
    #define documentDataFrame(compdata, columns=['latitude','longitude', 'value', 'unit', 'component'] )
    document = {
        'date': str(compdata['toTime'].iloc[0]), 
        'component': str(compdata['component'].iloc[0]),
        'data' : data.loc[data.component == component].to_json(orient='index'),
        'krige_data' : simplejson.dumps(krige_data.tolist())
    }
    # 'data' : compdata.to_json(orient='index'),
    #connect to db
    client = Cloudant(user, password, url=url, connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
    #store document
    document = db.create_document(document);
  return data.to_html()

#deletes entries, and old data
def clean_db():
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)

  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', '_id'])
  #transform the date column to datetime objects for date operations later
  fdocs['date'] = to_datetime(fdocs['date'])
  # Re-index dataframe
  fdocs = fdocs.reset_index(drop=True)
  #sort dataframe
  fdocs.sort_values(['date', 'component'])

  #get the components
  components = fdocs['component'].unique().tolist();

  #make a list of same size as components
  complist = [None]* len(components)
  for i in range(len(components)):
    #drop everything but relevant info
    tmp = fdocs.drop(fdocs[fdocs.component != components[i]].index)
    #sort them
    tmp = tmp.sort_values(['date'], ascending=[False])
    #re index the dataframe
    tmp = tmp.reset_index(drop=True)
    #put the dataframe into the list
    complist[i] = tmp;

  #how many days to keep data
  DAYS_OF_DATA_TO_KEEP = 14

  #go through all dataframes with data on each component
  for comp in complist:
    #go through each entry in the dataframe
    for i, row in comp.iterrows():
      #if there are more than 1 entry for the date remove it
      if comp[ row['date'] == comp.date ].count()[1] > 1:
        #remove entry from dataframe
        comp = comp.drop( comp[comp.date == row['date'] ].head(1).index )
        #remove it from the database
        db[row['_id']].delete()
        continue;
        
      
      #if there are entries older than DAYS_OF_DATA_TO_KEEP, remove them
      if((row['date'].today() - row['date']) > datetime.timedelta(DAYS_OF_DATA_TO_KEEP)):
        comp = comp.drop( comp[comp.date == row['date'] ].head(1).index )
        db[row['_id']].delete()
    print comp
  
  
  
  
@cron.interval_schedule(hours=3)
def job_function():
  kriging_plot()
  clean_db()

# Shutdown your cron thread if the web process is stopped
atexit.register(lambda: cron.shutdown(wait=False))
atexit.register(client.disconnect())

@app.route('/hello')
def hello():
  return 'hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
