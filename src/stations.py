from geopy.distance import vincenty
import pandas as pd

def city_station_locations(df_cities, df_stations, city_name):
    
    df_cities_clean = preprocess_cities(df_cities)
    id_city_sel = df_cities_clean.id.loc[df_cities['unique_name'] == city_name]
    stations_sel_loc_ser = df_stations['geometry'].loc[df_stations['city_id'] == id_city_sel.values[0]]
    frame = { 'location': stations_sel_loc_ser }
    stations_sel_loc = pd.DataFrame(frame)
    stations_sel_clean = preprocess_locations(stations_sel_loc)
    
    return stations_sel_clean


def preprocess_cities(df_cities):
    
    #transform country names into lowercase, no accent, blank space = _
    df_cities['country'] = df_cities['country'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_cities['country'] = df_cities['country'].str.replace(' ', '_')
    df_cities['country'] = df_cities['country'].str.lower()
    
    #transform city names into lowercase, no accent, blank space = _
    df_cities['name'] = df_cities['name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
    df_cities['name'] = df_cities['name'].str.replace(' ', '_')
    df_cities['name'] = df_cities['name'].str.lower()
    
    #preprocess the country_state column
    df_cities['country_state'] = df_cities['country_state'].str.replace(' ', '')
    df_cities['country_state'] = df_cities['country_state'].str.lower()
    
    #add name of country to all cities
    df_cities['unique_name'] = df_cities[['name', 'country']].apply(lambda x: '_'.join(x), axis=1)
    
    #check for remaining duplicated city names
    df_cities['is_duplicated'] = df_cities.duplicated('unique_name', keep = False)
    
    #add state to unique name, if still not unique
    df_cities.loc[df_cities['is_duplicated'] == True,'unique_name' ] = df_cities.loc[df_cities['is_duplicated'] == True][['unique_name', 'country_state']].apply(lambda x: '_'.join(x), axis=1)

    return df_cities


def preprocess_locations(sel_stations):
    
    #locations start with POINT, remove this text
    sel_stations.location = sel_stations.location.str.replace('POINT', '')
    sel_stations.location = sel_stations.location.str.replace('(', '')
    sel_stations.location = sel_stations.location.str.replace(')', '')

    #separate into 2 columns: longitude and latitude
    sel_stations[['longitude', 'latitude']]= sel_stations.location.apply(lambda x: pd.Series(str(x).split(" "))) 
    sel_stations['coords'] = sel_stations[['longitude', 'latitude']].apply(lambda x: ','.join(x), axis=1)
    return sel_stations
    

def find_nearest_station(sel_stations, listing_long, listing_lat):

    dist = []
    coords_listing = (listing_long, listing_lat)
    
    for row in sel_stations['coords'] :
        dist.append(vincenty(row, coords_listing).meters)
        
    return min(dist)

def create_closest_station_feature(df_ ,df_cities, df_stations, city_name):
    
    stations_sel_clean = city_station_locations(df_cities, df_stations, city_name)
    
    return df_.apply(lambda row: find_nearest_station(stations_sel_clean, row['longitude'], row['latitude']),axis=1)

    
    
    
    
    
    