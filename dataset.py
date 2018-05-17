from urllib2 import Request, urlopen, URLError
import json

import numpy as np
import math
from pandas import DataFrame, Series
from pandas.io.json import json_normalize
from pylab import *
import utm
from decimal import Decimal

# gps coordinates for map
# TOP LEFT      59.963886, 10.662291
# TOP RIGHT     59.963886, 10.887139
# BOTTOM RIGHT  59.873800, 10.887139
# BOTTOM LEFT   59.873800, 10.662291

# bottom left corner in GPS latitude and longitude
latitude_left_corner = 59.873800
longitude_left_corner = 10.662291

def data():
  # fetches data from nilu
  data = urlopen('https://api.nilu.no/aq/utd.json?areas=Oslo').read()
  # turns it into a json object
  jdata = json.loads(data)
  # puts it in a dataframe
  fdata = json_normalize(jdata)
  # picks out the wanted values
  fdata = DataFrame(fdata, columns=['latitude','longitude', 'value', 'unit', 'component', 'toTime'] )
  
  # add x, y columns to fdata
  fdata['x'] = 0
  fdata['y'] = 0
  
  for i, row in fdata.iterrows():  
    # add calculated x, y to dataframe and remove the offset (bottom left corner) to make a local coordinate x,y system
    fdata.loc[i, 'x'] = utm.from_latlon( row['latitude'], row['longitude'] )[0] - utm.from_latlon(latitude_left_corner, longitude_left_corner)[0];
    fdata.loc[i, 'y'] = utm.from_latlon( row['latitude'], row['longitude'] )[1] - utm.from_latlon(latitude_left_corner, longitude_left_corner)[1];
  
  return(fdata)