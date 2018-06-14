# Cloudant NoSQL imports
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
from cloudant.document import Document

# python flask, cf imports
from flask import Flask, render_template, send_file, url_for
import atexit
from apscheduler.scheduler import Scheduler
import cf_deployment_tracker # cloudfoundary

# system, io and other related imports
import os
import io

# imports for web
from urllib2 import urlopen
import json
import simplejson
import math
import re

# data structure and math imports
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pylab import *
from pandas import DataFrame, Series, to_datetime
from pandas.io.json import json_normalize, read_json
import datetime
import time

# import our other python files
from kriging_task import krige_task;
import dataset

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

# local coordinate system
X0, X1 = 0, RIGHT-LEFT
Y0, Y1 = 0, TOP-BOTTOM

# Scheduler for krige_task
cron = Scheduler(daemon=True)
cron.start()

app = Flask(__name__)

# On Bluemix, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8000))

# database varriables
db_name = 'luftkvalitet_db'
client = None
db = None

# test if you're on IBM cloud or running locally
if 'VCAP_SERVICES' in os.environ:
    # get environment varriables for database from your IBM cloud account
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
    # get environment varriables for database from a local vcap file
    with open('vcap-local.json') as f:
        vcap = json.load(f)
        print('Found local VCAP_SERVICES')
        creds = vcap['services']['cloudantNoSQLDB'][0]['credentials']
        user = creds['username']
        password = creds['password']
        url = 'https://' + creds['host']
        client = Cloudant(user, password, url=url, connect=True)
        db = client.create_database(db_name, throw_on_exists=False)
        
        
#Green, yellow, orange, red; color map
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
  
  #set the min max values according to whats unhealthy
  #max will be unhealthy value, and in turn be red on the generated image
  #we set values based on https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info&pollutant=no2 , Accsessed 11.05.2018
  if (doc['component'] == 'PM2.5'):
    maxvalue = 60;
    minvalue = 0;
  if (doc['component'] == 'PM10'):
    maxvalue = 85;
    minvalue = 0;
  if(doc['component'] == 'NO2'):
    maxvalue = 420;
    minvalue = 0; 
  #plot the figure
  fig, ax = subplots()
  ax.imshow(H, cmap=my_cmap, vmin=minvalue, vmax=maxvalue, origin='lower', interpolation='gaussian', alpha=0.7, extent=[X0, X1, Y0, Y1])
  ax.axis('off')
  
  #set proper image size and extent to plot
  fig.set_size_inches(5.95, 5)
  imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  fig.savefig(buffr, format='svg', bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  
  #go to start of file
  buffr.seek(0)
  
  #disconnect from db
  client.disconnect()
  
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
  
  #set standard minmax, these will be changed based on component
  maxvalue = 100;
  minvalue = 0;
  
  #set the min max values according to whats unhealthy
  #max will be unhealthy value, and in turn be red on the generated image
  #we set values based on https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info&pollutant=no2 , Accsessed 11.05.2018
  if (doc['component'] == 'PM2.5'):
    maxvalue = 60;
    minvalue = 0;
  if (doc['component'] == 'PM10'):
    maxvalue = 85;
    minvalue = 0;
  if(doc['component'] == 'NO2'):
    maxvalue = 420;
    minvalue = 0; 

  fig, ax = subplots()
  #generate contour plot
  CS = plt.contour(H,vmin=minvalue, vmax=maxvalue, extend='max',  colors='royalblue', origin='lower', alpha=1, extent=[X0, X1, Y0, Y1])
  #put the contour into the figure
  plt.clabel(CS, inline=10, fontsize=4 , fmt='%0.1f')
  
  #take off the axises and set dpi (pixel density)
  ax.axis('off')
  fig.dpi=400
  #set size
  fig.set_size_inches(5.95, 5)
  #get the extent that should be ploted
  imgextent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  #save it to the buffer 
  fig.savefig(buffr, dpi = 400, bbox_inches=imgextent, transparent=True, pad_inches=0, frameon=None)
  #set buffer to point at the start.
  buffr.seek(0)
  
  #disconnect from db
  client.disconnect()
  
  #send the buffer with the contour plot
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
  
  #set standard minmax, these will be changed based on component
  maxvalue = 100;
  minvalue = 0;
  
  #set the min max values according to whats unhealthy
  #max will be unhealthy value, and in turn be red on the generated image
  #we set values based on https://uk-air.defra.gov.uk/air-pollution/daqi?view=more-info&pollutant=no2 , Accsessed 11.05.2018
  if (doc['component'] == 'PM2.5'):
    maxvalue = 60;
    minvalue = 0;
  if (doc['component'] == 'PM10'):
    maxvalue = 85;
    minvalue = 0;
  if(doc['component'] == 'NO2'):
    maxvalue = 420;
    minvalue = 0; 
  
  buffr = io.BytesIO()
  fig, ax = subplots()    
  fig.dpi=400
  ax.imshow(H, cmap=my_cmap, vmin=minvalue, vmax=maxvalue, origin='lower', interpolation='nearest', alpha=0.7, extent=[X0, X1, Y0, Y1])

  sc = ax.scatter( z.x, z.y, cmap=my_cmap,vmin=minvalue, vmax=maxvalue, c=z.value, linewidths=0.75, s=50 )
  plt.colorbar(sc)

  fig.suptitle('component: ' + str(z['component'].iloc[0]) + ', date: ' + str(date), fontsize=14)
  fig.savefig(buffr, dpi = 400)
  
  buffr.seek(0)
  #disconnect from db
  client.disconnect()
  return send_file(buffr, mimetype='image/png')
  
@app.route('/')
def show_data():
  #connect to the db
  client = Cloudant(user, password, url=url, connect=True)
  db = client.create_database(db_name, throw_on_exists=False)
  
  #get all docs
  docs = list(map(lambda doc: doc, db) )
  #put them into a dataframe
  fdocs = json_normalize(docs);
  fdocs = DataFrame(fdocs, columns=['date', 'component', 'data', '_id'])
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
    #drop duplicates 
    tmp = tmp.drop_duplicates(subset=['date'], keep='first', inplace=False);
    #sort them
    tmp = tmp.sort_values(['date'], ascending=[False])
    #re index the dataframe
    tmp = tmp.reset_index(drop=True)
    #put the dataframe into the list
    complist[i] = tmp;
  #disconnect from db
  client.disconnect()
  return render_template('index.html', entries = complist, data = data);
  


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
    #drop duplicates 
    tmp = tmp.drop_duplicates(subset=['date'], keep='first', inplace=False);
    #sort them
    tmp = tmp.sort_values(['date'], ascending=[False])
    #re index the dataframe
    tmp = tmp.reset_index(drop=True)
    #put the dataframe into the list
    complist[i] = tmp;
  #disconnect from db
  client.disconnect()
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
  #disconnect from db
  client.disconnect()
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

  #how many days to keep data, each day is about 20KB worth of data
  #keep this number under 25, as long as you are on a lite/trial account
  #cloudant lite wont let you fetch more than 25 days worth of data
  DAYS_OF_DATA_TO_KEEP = 20

  #rate limit, changes pr second to the database
  #cloudant lite has a rate limit of 10 changes pr second.
  RATE_LIMIT = 10
  
  #changes before rate limit is reached
  changes = 0;
  
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
        changes += 1;
        continue;
        
      #if there are entries older than DAYS_OF_DATA_TO_KEEP, remove them
      if((row['date'].today() - row['date']) > datetime.timedelta(DAYS_OF_DATA_TO_KEEP)):
        comp = comp.drop( comp[comp.date == row['date'] ].head(1).index )
        db[row['_id']].delete()
        changes += 1;
      
      #sleep so you dont breach the rate limit
      if changes < RATE_LIMIT-2:
        time.sleep(1);
        changes = 0;
        
  #disconnect from db
  client.disconnect()
  
#set the time interval to run kriging and clean_db
@cron.interval_schedule(hours=3)
def job_function():
  kriging_plot()
  clean_db()

# Shutdown the cron thread if the web process is stopped (used to shutdown thread cycling the krige_task)
atexit.register(lambda: cron.shutdown(wait=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)