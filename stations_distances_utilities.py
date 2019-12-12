import json
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pandas.io.json import json_normalize 



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

def dist_to_nearest_station(stations_coord, listing_coord):
    
    min_distances = []
    idx = 0
    for listing in listing_coord:
        dist_tmp = []
        idx += 1
        print('Running.... ' + str(np.ceil((idx/len(listing_coord))*100)) + '%', end='\r')
        for station in stations_coord:
            dist_tmp.append(geodesic(station, listing).meters)
        
        min_distances.append(min(dist_tmp))
    print('\n')
    return min_distances
