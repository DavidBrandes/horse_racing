import numpy as np


EARTH_RADIUS = 6371000  # in meters


def to_xy(lat, long, origin_lat, origin_long): 
    deg_to_rad = np.pi / 180
    
    lat = lat * deg_to_rad
    long = long * deg_to_rad
    origin_lat = origin_lat * deg_to_rad
    origin_long = origin_long * deg_to_rad
      
    x = EARTH_RADIUS * (long - origin_long) * np.cos(origin_lat)
    y = EARTH_RADIUS * (lat - origin_lat)
  
    return np.stack([x, y], axis=-1)