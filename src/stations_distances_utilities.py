import json
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pandas.io.json import json_normalize 
from math import radians, cos, sin, asin, sqrt




def geojson_to_dataframe(file_path):
    with open(file_path) as data_file:    
        data = json.load(data_file)

    df = pd.DataFrame()
    for i in data['features']:
        df = pd.concat([df, json_normalize(i)], sort = True)

    df = df.reset_index(drop = True)
    df.columns = [s.split('.')[-1] for s in df.columns.values]
    return df

def get_listing_coord(listing_df):
    #df = pd.read_csv(listing_file_path, low_memory = False, header = 0)
    listing_coord = [(listing_df.longitude.values[i], listing_df.latitude.values[i]) for i in range(0, len(listing_df.longitude.values))]
    return listing_coord

def get_station_coord(stations_df):
    #stations = pd.read_csv(station_file_path, header = 0)
    
    stations_long = stations_df.coordinates.astype(str).apply(lambda x: x.split(',')[0].split('[')[1])
    stations_lat = stations_df.coordinates.astype(str).apply(lambda x: x.split(',')[-1].split(']')[0])
    stations_coord = [(stations_long[i], stations_lat[i]) for i in range(0, len(stations_long))]
    return stations_coord

def haversine_distance(point, list_of_coord):
    '''
    sources:
    implementation:   https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    explanations:     https://janakiev.com/blog/gps-points-distance-python/
    original formula: https://en.wikipedia.org/wiki/Haversine_formula
    '''
    
    # convert decimal degrees to radians 
    point = np.deg2rad([float(i) for i in point])
    list_of_long = np.deg2rad([float(i[0]) for i in list_of_coord])
    list_of_lat = np.deg2rad([float(i[1]) for i in list_of_coord])

    # haversine formula 
    dlon = list_of_long - point[0] 
    dlat = list_of_lat - point[1]
    a = np.sin(dlat/2)**2 + np.cos(point[1]) * np.cos(list_of_lat) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    R = 6.3781e6 # Radius of earth in meters
    return c * R

def geodesic_distance(point, list_of_coord):
    dists = []
    for coord in list_of_coord:
        dists = np.append(dists, geodesic((point[1], point[0]), (coord[1], coord[0])).meters)
    return dists

def dist_to_nearest_station_haversine(stations_coord, listing_coord):
    
    min_distances = []
    idx = 0
    for listing in listing_coord:
        dist_tmp = []
        idx += 1
        print('Running Haversine distances calculation.... ' + str(np.ceil((idx/len(listing_coord))*100)) + '%', end='\r')
        #for station in stations_coord:
        #    dist_tmp.append(geodesic(station, listing).meters)
        dist_tmp = np.append(dist_tmp, haversine_distance(listing, stations_coord))
        min_distances.append(min(dist_tmp))
    print('\n')
    return min_distances


def dist_to_nearest_station_geodesic(stations_coord, listing_coord):
    
    min_distances = []
    idx = 0
    for listing in listing_coord:
        dist_tmp = []
        idx += 1
        print('Running geodesic distances calculation.... ' + str(np.ceil((idx/len(listing_coord))*100)) + '%', end='\r')
        #for station in stations_coord:
        #    dist_tmp.append(geodesic(station, listing).meters)
        dist_tmp = np.append(dist_tmp, geodesic_distance(listing, stations_coord))
        min_distances.append(min(dist_tmp))
    print('\n')
    return min_distances


