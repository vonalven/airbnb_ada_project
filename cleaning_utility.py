import os
import time
import pydot
import folium
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotly.graph_objects as go
from folium.plugins import MiniMap
from IPython.core.display import display
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti, utils
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


class CleaningUtility():
    """
    CleaningUtility is a class defining tools to clean the airbnb data set

    Attributes:
    -----------
    - (Empty class)

    Methods:
    -----------
    bool_to_int(df, columns_names)
        convert string bool ('t'/'f') to integer
        
    host_activity_period(df, column)
        computes the host period of activity from the date in column "column" to 2019
        
    list_to_number_of_services(df, columns, drop_col = True)
        convert list (array) of services to number of elements in the list
        
    convert_to_one_hot_label(df, columns)
        convert categorical to one-hot-label (one column > multiple columns)
    
    replace_nan_by_values(df, columns_list, values_list)
        replace NaNs in columns of columns_list by values in values_list
        
    format_price(df, columns_list)
        format the price contained in columns of columns_list array. For example $1,000 becomes 1000
        
    format_rate(df, columns_list)
        format the rates contained in columns of colums_list array. For example 90% becomes 90
    
    string_to_id(df, df_column_name, reference_strings, reference_IDs)
        converts categorical strings to numerical
        
    prices_per_person(df, price_columns, nb_persons_column)
        computes the price per person
        
    select_numeric_column_only(df)
        determines which columns are entirely numeric in the dataframe and selects them
    """
    
    def __init__(self):
        """
        Empty constructor
        """
        
        pass
    
    def bool_to_int(self, df, columns_names):
        """converts string bool ('t'/'f') to integer
        
        Parameters:
        -----------
        df : pandas DataFrame
        columns_names : array of string
            names of the columns where conversion need to be applied

        Returns:
        -----------
        df : pandas DataFrame
        """
        
        print('> Running bool_to_int...')
        map_bool = {'t':1, 'f':0, 'nan':np.nan}
        dict_replace = {}
        for col in columns_names:
            dict_replace.update({col : map_bool})

        df[columns_names] = df[columns_names].astype('str').replace(dict_replace)

        # test conversion
        #all_is_num = df[columns_names].notnull().all()
        #unique_values = np.unique(df[columns_names].values)
        #if all_is_num.all() & (set([0, 1]) == set(unique_values)):
        #    print('All boolean correctly converted')
        #elif np.isnan(unique_values).any():
        #    print('All boolean correctly converted but some nan are present\n')

        return df

    def host_activity_period(self, df, column):
        """computes the host period of activity from the date in column "column" to 2019
        
        Parameters:
        -----------
        df : pandas DataFrame
        column : string
            column name where the tomporal data are stored

        Returns:
        -----------
        df : pandas DataFrame
        """
        
        print('> Running host_activity_period...')
        start_year = pd.DatetimeIndex(df[column]).year
        start_month = pd.DatetimeIndex(df[column]).month
        current_month = 10
     
        period_activity_months = current_month + 12*(2019-1-start_year) + (12 - start_month)

        df[column] = period_activity_months

        unique_values = df[column].unique()
        if np.isnan(unique_values).any():
            print('All activities periods extracted but some nan are present\n')
        return df

    def list_to_number_of_services(self, df, columns, drop_col = True):
        """converts list (array) of services to number of elements in the list
        Parameters:
        -----------
        df : pandas DataFrame
        columns : array of string
            columns names where the lists to be converted are stored
        drop_col : bool
            if True, the original columns in parameter "columns" are dropped after conversion

        Returns:
        -----------
        df : pandas DataFrame
        """
        
        print('> Running list_to_number_of_services...')
        for col in columns:
            df['number_of_' + col] = df[col].apply(lambda x: len(x.replace('"', '').replace('{', '').replace('}', '').split(',')))
            if drop_col:
                df = df.drop(col, axis = 1) 
        return df

    def convert_to_one_hot_label(self, df, columns):
        """converts categorical to one-hot-label (one column > multiple columns)
        Parameters:
        -----------
        df : pandas DataFrame
        columns : array of string
            columns names where the categorical data to be converted are located

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running convert_to_one_hot_label...')
        df = pd.get_dummies(df, prefix = columns, columns = columns)
        return df

    def replace_nan_by_values(self, df, columns_list, values_list):
        """replaces NaNs in columns of columns_list by values in values_list
        
        Parameters:
        -----------
        df : pandas DataFrame
        colums_list : array of string
            columns names where the replacement must be applied
        values_list : array of values
            list of values used to replace NaNs. One value must be specified for each column in columns_list

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running replace_nan_by_values...')
        value_replace = {}
        for col, val in zip(columns_list, values_list):
            value_replace.update({col:val})
        df[columns_list] = df[columns_list].fillna(value = value_replace)
        return df

    def format_price(self, df, columns_list):
        """formats the price contained in columns of columns_list array. For example $1,000 becomes 1000
        
        Parameters:
        -----------
        df : pandas DataFrame
        columns_list : array of string
            columns names where the prices to be converted are located

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running format_price...')
        for col in columns_list:
            # replace , in price by '' and remove $ if present (ex $658,000.0)
            df[col] = df[col].apply(lambda x: x.replace(',', '').split('$')[1] if (type(x) is str and '$' in x) else x)
        return df

    def format_rate(self, df, columns_list):
        """formats the rates contained in columns of colums_list array. For example 90% becomes 90
        
        Parameters:
        -----------
        df : pandas DataFrame
        columns_list : array of string
            columns names where the rates to be converted are located

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running format_rate...')
        for col in columns_list:
            df[col] = df[col].apply(lambda x: x.split('%')[0] if (type(x) is str and '%' in x) else x)
        return df

    # convert strings in df column df_column_name to id in reference_IDs associated to the reference_strings
    def string_to_id(self, df, df_column_name, reference_strings, reference_IDs):
        """converts categorical strings to numerical (to classes)
        
        Parameters:
        -----------
        df : pandas DataFrame
        df_column_name : string
            name of the column in df where the strings to be converted are present
        reference_strings : array of string
            list of categorical strings found in df_column_name
        reference_IDs : array of string
            numerical label associated to each element in reference_strings

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running string_to_id...')
        id_replace = {}
        for ref_s, ref_id in zip(reference_strings, reference_IDs):
            id_replace.update({ref_s:ref_id})
        df[df_column_name] = df[df_column_name].replace(id_replace)
        return df

    def prices_per_person(self, df, price_columns, nb_persons_column):
        """computes the price per person
        
        Parameters:
        -----------
        df : pandas DataFrame
        price_columns : array of string
            columns names where the prices are located
        nb_persons_column : string
            column name where the total number of person allowed for the total reference price is located

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running prices_per_person...')
        df[price_columns] = df[price_columns].astype('float64')
        df[nb_persons_column] = df[nb_persons_column].astype('float64')
        for p_c in price_columns:
            df[p_c + '_per_person'] = df[p_c] / df[nb_persons_column]
        return df

    def select_numeric_column_only(self, df):
        """determines which columns are entirely numeric in the dataframe and selects them
        Parameters:
        -----------
        df : pandas DataFrame

        Returns:
        -----------
        df : pandas DataFrame

        """
        
        print('> Running select_numeric_column_only...')
        df = df[df.notnull().any(axis=1)]
        return df
