import pickle
from feature_tools import *



def perform_rf_simple(df_full, df_success_metrics, metric, random_seed, tune = False, plot_res = [True, True, False, False],
                      saveFig = [False, False, False, False], save_tag = None, save_destination = None):
    '''
    takes the DataFrame containing features and metrics as argument, removes all metrics except one
    which becomes then the 'label' and finally performs the random forest on these datas
    '''
    df = df_full.copy().drop(columns = df_success_metrics.columns.difference([metric]))
    f = FeaturesTools(df, metric, random_seed = random_seed)
    processed_feat = f.preprocess_features(preprocessing_type = 'standardize_robust')
    f = FeaturesTools(processed_feat, metric, random_seed = random_seed)
    df_importance, df_err, rf = f.randomForestAnalysis(plotResults = plot_res,
                                                       tuneModelParameters = tune, saveFigures = saveFig, 
                                                       saveTag = save_tag, saveLocation = save_destination)
    
    return df_importance, df_err, rf



def build_sankey_input_df(list_of_df, list_of_targets, column_weights_name, column_features_name, weight_treshold = None):
    '''
    weight_treshold allows to introduce a weight for the display of the connections. Only connections with a 
    weight > weight_treshold will be shown!
    '''
    df_sankey = pd.DataFrame()
    features_names = []

    for df, label in zip(list_of_df, list_of_targets):
        
        if weight_treshold is not None:
            df = df[df[column_weights_name] > weight_treshold]
            
        features_names = df[column_features_name].tolist()
        df_connections_tmp = pd.DataFrame({'feature_source': features_names,
                                        'feature_target': [label]*len(features_names),
                                        'contribution'  : df[column_weights_name].tolist()})
        df_sankey = pd.concat([df_sankey, df_connections_tmp])

    
    all_sources = df_sankey.feature_source.unique().tolist()
    all_targets = df_sankey.feature_target.unique().tolist()
    labels = all_sources + all_targets
    ids    = np.arange(0, len(labels))
    replace_dict = {}
    for ref_s, ref_id in zip(labels, ids):
        replace_dict.update({ref_s:ref_id})
    df_sankey['source'] = df_sankey['feature_source'].replace(replace_dict)
    df_sankey['target'] = df_sankey['feature_target'].replace(replace_dict)
    labels = labels + [np.nan]*(df_sankey.shape[0]-len(labels))
    df_sankey['labels'] = labels    


    nb_colors = len(all_targets)
    colors = sns.color_palette('Set1', nb_colors)

    # add souces/targets boxs colors
    color_nodes = ['#808080'] * len(all_sources) + colors.as_hex()
    color_nodes = color_nodes + [np.nan]*(df_sankey.shape[0]-len(color_nodes))
    df_sankey['color'] = color_nodes

    f = FeaturesTools()

    # add trajectories colors
    colors_trajectories = []
    for target, color in zip(df_sankey.target.unique(), colors):
        df_tmp = df_sankey[df_sankey['target'] == target]
        target_color_scale   = sns.light_palette(color, 4, reverse = True).as_hex()

        # convert color palette to string format style: rgba(253, 253, 253, 0.5)
        #tmp = []
        #for jj in target_color_scale:
            #tmp = np.append(tmp, 'rgb' + str(tuple(int(np.ceil(i * 255)) for i in jj[0:3]) + (jj[3], )))
        #target_color_scale = tmp
        target_mapped_colors = f.map_to_color(target_color_scale, df_tmp.contribution)
        colors_trajectories  = np.append(colors_trajectories, target_mapped_colors)
    df_sankey['linkColor'] = colors_trajectories
    
    return df_sankey
    


def extract_from_all_datasets(target_hosts_ids, target_hosts_names, data_location, list_of_listing_files, list_of_comments_files, search_name):
    '''
    1) Extract from the files names in list list_of_listing_files all the rows that are associated to the host ids 
       specified in the list target_hosts_ids
    2) From the obtained dataset, extract the unique id. The id in the listing_detailed.csv files correspond to the 
       listing_id in the reviews.csv file. The obtained list is called id_listings and contains all the listings ids
       that belong to the target hosts
    3) Extract from the files names in list_of_comments_files all the rows (comments info) that are associated to the 
       listings in the list id_listings previously created. By this way we extract all the comments for the listings
       that are owned by the target hosts
    4) Save the 2 created DataFrame. In the saving name the search_name key is included. This allow to use this function
       multiple time without overwriting the output files. For example search_name can be 'international' if we are
       extracting info for the hosts present in more than one city (> international hosts)
    '''
    
    df_all_hosts = pd.DataFrame()
    df_all_hosts_comments = pd.DataFrame()
    
    
    t0 = time.time()
    
    col_names_listings = None
    idx_advancment     = 0
    for file_detailed in list_of_listing_files:

        idx_advancment    += 1
        city_name_detailed = file_detailed.split('_')[1]
        print('> ' + str(int(idx_advancment)) + '/' + str(int(len(list_of_listing_files))) + ' : Analyzing listings for city ' + city_name_detailed + '.........................', end = '\r')

        tmp_detailed = pd.read_csv(data_location + '/' + file_detailed, low_memory = False)
        
        # keep common columns only!
        if idx_advancment == 1:
            col_names_listings = tmp_detailed.columns.tolist()
        common_columns = list(set(col_names_listings).intersection(tmp_detailed.columns.tolist()))
        diff_columns   = list(set(col_names_listings).symmetric_difference(set(tmp_detailed.columns.tolist())))
        if len(diff_columns) > 0:
            print('\n\nThe following columns were not found in the listings dataset for city %s:\n'%(city_name_detailed))
            print(diff_columns)
            print()
            tmp_detailed   = tmp_detailed[common_columns]
        
        # keep only rows corresponding to international hosts
        tmp2_detailed  = tmp_detailed[(tmp_detailed['host_id'].isin(target_hosts_ids)) & (tmp_detailed['host_name'].isin(target_hosts_names))]
        if tmp2_detailed.shape[0] > 0:
            # if there is some columns missing in the current data-set and if some rows of it have to be concatenate to the 
            # common data-set, keep only common columns!
            if len(diff_columns) > 0:
                df_all_hosts = df_all_hosts[common_columns]
            df_all_hosts = pd.concat([df_all_hosts, tmp2_detailed], sort = False)
        
    
    # now take all the listings id (=id in detailed csv file) and take all the correaponding comments
    # The ids of the target hosts are:
    id_listings = df_all_hosts.id.unique().tolist()
    
    col_names_comments = None
    idx_advancment = 0
    for file_comment in list_of_comments_files:

        idx_advancment += 1
        city_name_comment = file_comment.split('_')[1]  
        print('> ' + str(int(idx_advancment)) + '/' + str(int(len(list_of_comments_files))) + ' : Analyzing comments for city ' + city_name_comment + '.........................', end = '\r')
        
        tmp_comments = pd.read_csv(data_location + '/' + file_comment, low_memory = False)
        
        # keep common columns only!
        if idx_advancment == 1:
            col_names_comments = tmp_comments.columns.tolist()
        common_columns = list(set(col_names_comments).intersection(tmp_comments.columns.tolist()))
        diff_columns   = list(set(col_names_comments).symmetric_difference(set(tmp_comments.columns.tolist())))
        if len(diff_columns) > 0:
            print('\n\nThe following columns were not found in the comments dataset for city %s:\n'%(city_name_comment))
            print(diff_columns)
            print()
        tmp_comments   = tmp_comments[common_columns]
        
        # keep only rows corresponding to comments for listings owned by international hosts
        tmp2_comments = tmp_comments[tmp_comments['listing_id'].isin(id_listings)]

        if tmp2_comments.shape[0] > 0:
            # if there is some columns missing in the current data-set and if some rows of it have to be concatenate to the 
            # common data-set, keep only common columns!
            if len(diff_columns) > 0:
                df_all_hosts_comments = df_all_hosts_comments[common_columns]
            tmp2_comments['city'] = [city_name_comment]*tmp2_comments.shape[0]
            df_all_hosts_comments = pd.concat([df_all_hosts_comments, tmp2_comments], sort = False)
    
    # save results as .csv
    print('\n')
    print('All the DataSet is collected!\nElapsed time.... %f [seconds]\n'%(time.time()-t0))
    print('Saving to .csv ...\n')
    df_all_hosts.to_csv('df_all_hosts' + '_' + search_name + '.csv', index = False)
    df_all_hosts_comments.to_csv('df_all_hosts_comments' + '_' + search_name + '.csv', index = False)
    print('All files saved!\n')



def save_list(save_name, save_destination, list_object):
    with open(save_destination + '/' + save_name + '.txt', 'wb') as fp: 
           pickle.dump(list_object, fp)



def read_list(file_name, file_location):
    with open(file_location + '/' + file_name, 'rb') as fp:
        list_object = pickle.load(fp)
    return list_object
   