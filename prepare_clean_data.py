from cleaning_utility import *

def buildSuccessMetricsDf(df_listings, df_sentiment):
    df_success_metrics = pd.merge(df_listings[['id', 'review_scores_rating', 'reviews_per_month']].astype('float64'),
                  df_sentiment, left_on='id', right_on='listing_id')
    # change index to id
    df_success_metrics = df_success_metrics.set_index('id').drop('listing_id', axis = 1)

    return df_success_metrics

def cleanAndMergeData(df_listings, df_success_metrics, use_neigh = True):
    if use_neigh:
        features_ensemble       = ['id','host_since', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
                            'host_verifications', 'host_has_profile_pic', 'host_identity_verified', 'neighbourhood_cleansed',
                            'property_type', 'room_type', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee',
                            'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
                            'instant_bookable', 'cancellation_policy', 'dist_nearest_station']
    else:
        features_ensemble       = ['id','host_since', 'host_response_rate', 'host_is_superhost', 'host_total_listings_count', 
                            'host_verifications', 'host_has_profile_pic', 'host_identity_verified',
                            'property_type', 'room_type', 'bed_type', 'amenities', 'price', 'security_deposit', 'cleaning_fee',
                            'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights', 'number_of_reviews',
                            'instant_bookable', 'cancellation_policy', 'dist_nearest_station']
    
    # features are seperated depending on their types to be prepared before being used for ML
    date_features           = 'host_since'
    bool_features           = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    list_features           = ['host_verifications', 'amenities']
    price_features          = ['price', 'security_deposit', 'cleaning_fee', 'extra_people']
    if use_neigh:
        string_features         = ['property_type', 'cancellation_policy','room_type', 'bed_type','neighbourhood_cleansed']
    else:
        string_features         = ['property_type', 'cancellation_policy','room_type', 'bed_type']
    rate_features           = ['host_response_rate']
    replace_nan_features    = ['host_response_rate', 'host_is_superhost', 'host_total_listings_count', 'security_deposit', 'cleaning_fee', 'number_of_amenities']
    replace_values          = [0, 0, 1, 0, 0, 0]
    

    
    # create an instance of the class CleaningUtility()
    cu = CleaningUtility()
    ##############################################################
    ############# using the class CleaningUtility(), #############
    ############# features are prepared to be used   #############                   
    ##############################################################
    df_features = df_listings[features_ensemble].copy()
    df_features = cu.bool_to_int(df_features, bool_features)
    df_features = cu.host_activity_period(df_features, date_features)
    df_features = cu.list_to_number_of_services(df_features, list_features)
    df_features = cu.format_price(df_features, price_features)
    df_features = cu.format_rate(df_features, rate_features)
    df_features = cu.replace_nan_by_values(df_features, replace_nan_features, replace_values)
    df_features = cu.convert_to_one_hot_label(df_features, string_features)
    df_features = cu.prices_per_person(df_features, price_features[0:-1], 'guests_included')

    ##############################################################
    ## now that all the data are numeric, convert all to float64 #
    ##############################################################
    cols              = df_features.columns
    df_features[cols] = df_features[cols].apply(pd.to_numeric, errors = 'raise')

    ##############################################################
    ############# keep only rows with non-nan values #############
    ##############################################################
    tmp         = df_features.shape[0]
    df_features = cu.select_numeric_column_only(df_features)
    df_features = df_features.dropna()

    ##############################################################
    ############## remove rows with infinite values ##############
    ##############################################################
    df_features = cu.remove_rows_with_infinite(df_features)


    ##############################################################
    ##################### print cleaning info ####################
    ##############################################################
    print('\nNumber of rows    before data set cleaning:       %.0f'%(df_listings.shape[0]))
    print(  'Number of rows    after data set cleaning:        %.0f'%(tmp))
    print(  'Number of rows    after removal of rows with nan: %.0f'%(df_features.shape[0]))
    print(  'Number of columns before data set cleaning:       %.0f'%(df_listings.shape[1]))
    print(  'Number of columns initially selected:             %.0f'%len(features_ensemble))
    print(  'Number of columns after data set cleaning:        %.0f'%(df_features.shape[1]))


    ##############################################################
    ####################### clean df_metric ######################
    ##############################################################
    df_metrics = cu.remove_by_threshold(df_success_metrics, 'reviews_per_month', 30)
    df_metrics = df_success_metrics.dropna(axis = 0)

    ##############################################################
    ############# merge features and metric datasets #############
    ##############################################################
    # merge to have metrics and features in the same dataFrame
    df_cleaned_merged = pd.merge(df_metrics, df_features, left_on = 'id', right_on = 'id')

    # set 'id' as the index of the dataFrame
    df_cleaned_merged = df_cleaned_merged.set_index('id')

    return df_cleaned_merged
