from urllib2 import Request, urlopen, URLError
import json

import numpy as np
import math
from pandas import DataFrame, Series
from pandas.io.json import json_normalize
from pylab import *
import utm
from decimal import Decimal

def data():
  # fetches data from nilu
  data = urlopen('https://api.nilu.no/aq/utd.json?areas=Oslo').read()
  # turns it into a json object
  jdata = json.loads(data)
  # puts it in a dataframe
  fdata = json_normalize(jdata)
  # picks out the wanted values
  fdata = DataFrame(fdata, columns=['latitude','longitude', 'value', 'unit', 'component', 'toTime'] )
  
  # print(fdata)
  
  fdata['x'] = 0
  fdata['y'] = 0
  #fdata['index'] = fdata.index;
  
  # gps coordinates for map
  # TOP LEFT      lat2 = 59.963886#,  long2 = 10.662291
  # TOP RIGHT     
  lat2 = 59.963886
  long2 = 10.887139
  # BOTTOM RIGHT  59.873800, 10.887139
  # BOTTOM LEFT   59.873800, 10.662291
  
  # bottom left:
  lat1 = 59.873800
  long1 = 10.662291
  
  for i, row in fdata.iterrows():  
    # add calculated x, y to dataframe
    fdata.loc[i, 'x'] = utm.from_latlon( row['latitude'], row['longitude'] )[0] - utm.from_latlon(lat1, long1)[0];
    fdata.loc[i, 'y'] = utm.from_latlon( row['latitude'], row['longitude'] )[1] - utm.from_latlon(lat1, long1)[1];
  #fdata = fdata.drop(fdata[fdata.component != 'PM10'].index)
  return(fdata)
