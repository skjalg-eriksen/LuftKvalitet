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
from PIL import Image

import atexit
from apscheduler.scheduler import Scheduler

import numpy as np
from pandas.io.json import json_normalize, read_json
from pylab import *
import numpy as np
from pandas import DataFrame, Series
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
        

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/img/<_id>')
def get_img(_id):
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
    
    X0, X1 = 0, RIGHT-LEFT
    Y0, Y1 = 0, TOP-BOTTOM

    #connect to the db
    client = Cloudant(user, password, url=url, connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
    #get the document from the db
    doc = db[ _id ]
    H = simplejson.loads(doc['krige_data'])
    buffr = io.BytesIO()
    
    fig, ax = subplots()
    ax.imshow(H, cmap=my_cmap, origin='lower', interpolation='gaussian', alpha=0.7, extent=[X0, X1, Y0, Y1])
    ax.axis('off')
    fig.dpi=400

    fig.set_size_inches(5.95, 5)
    imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(buffr, dpi = 400, bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
    buffr.seek(0)
    
    return send_file(buffr, mimetype='image/png')
    

@app.route('/info/<_id>')
def get_info(_id):
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
    
    buffr = io.BytesIO()
    fig, ax = subplots()    
    fig.dpi=400
    ax.imshow(H, cmap=my_cmap, origin='lower', interpolation='nearest', alpha=0.7, extent=[X0, X1, Y0, Y1])

    sc = ax.scatter( z.x, z.y, cmap=my_cmap, c=z.value, linewidths=0.75, s=50 )
    plt.colorbar(sc)

    fig.suptitle('component: ' + str(z['component'].iloc[0]) + ', date: ' + str(date), fontsize=14)
    fig.savefig(buffr, dpi = 400)
    
    buffr.seek(0)
    
    return send_file(buffr, mimetype='image/png')

@app.route('/show_data', methods=['GET'])
def show_data():
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  
  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', '_id'])
  fdocs = fdocs.reset_index(drop=True)
  fdocs.sort_values(['date', 'component'])
  #get the components
  components = fdocs['component'].unique().tolist()
  
  #get the requested component
  if( request.args.get('selected_component') ):
    selected_component = request.args.get('selected_component')
  else:
    selected_component = "PM10"
  
  #drop what we dont need.
  fdocs = fdocs.drop(fdocs[fdocs.component != selected_component].index)
  fdocs = fdocs.sort_values(['date'], ascending=[False])
  fdocs = fdocs.reset_index(drop=True)
  
  if( request.args.get('index') ):
    i = int(request.args.get('index'))
    selected_id = fdocs.loc[i, '_id']
    selected_date = fdocs.loc[i, 'date']
    
  else:
    i = 0
    selected_id = fdocs.loc[i, '_id']
    selected_date = fdocs.loc[i, 'date']
  
  
  return render_template('show.html', 
  components = components,
  index = i,
  _id = selected_id,
  date = selected_date,
  component = selected_component,
  fdocs = fdocs)
  
@app.route('/all_entries', methods=['GET'])
def all_entries():
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  
  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', '_id'])
  fdocs = fdocs.reset_index(drop=True)
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
 
    
  return render_template('entries.html', entries = complist);
  
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
  date =  str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
  data = dataset.data()
  data.sort_values(by=['component'])
  #data.set_index(keys=['component'], drop=False,inplace=True)
  names = data['component'].unique().tolist()
  
  for component in names:
    compdata = data.loc[data.component == component]
    if(4 >len(compdata.index)):
      continue;
    
    krige_data = krige_task(compdata, date)
    
    #define document
    document = {
        'date': date, 
        'component': str(compdata['component'].iloc[0]),
        'data' : compdata.to_json(orient='index'),
        'krige_data' : simplejson.dumps(krige_data.tolist())
    }
    #connect to db
    client = Cloudant(user, password, url=url, connect=True)
    db = client.create_database(db_name, throw_on_exists=False)
    #store document
    document = db.create_document(document);
    
  return "done.."
  
@cron.interval_schedule(hours=3)
def job_function():
  kriging_plot()
  

# Shutdown your cron thread if the web process is stopped
atexit.register(lambda: cron.shutdown(wait=False))
atexit.register(client.disconnect())

@app.route('/hello')
def hello():
  return 'hello World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)
