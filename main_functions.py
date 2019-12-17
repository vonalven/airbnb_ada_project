import pickle
import plotly.express as px
from feature_tools import *
from RBO.rankedlist import RBO
import plotly.graph_objects as go
from scipy.ndimage.measurements import label



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


def pairwise_similarity_clustered(df_all_importances, first_n_ranks, success_metrics_list):
    all_cities = df_all_importances.city.unique().tolist()
    df_rbo_similarity_list = []
    
    if success_metrics_list == 'multi':
        success_metrics_list = [success_metrics_list]
        
    for metric in success_metrics_list:
        df_rbo_similarity = pd.DataFrame()
        for current_city in all_cities:
            current = df_all_importances.loc[df_all_importances.city == current_city]
            if metric is not 'multi':
                current = current.loc[current.metric == metric]
            current = current.iloc[0:first_n_ranks, :]

            rbo_scores_current = {current_city : 1.0}
            for other_cities in [x for x in all_cities if x != current_city]:
                df_other_city = df_all_importances.loc[df_all_importances.city == other_cities]
                if metric is not 'multi':
                    df_other_city = df_other_city.loc[df_other_city.metric == metric]
                df_other_city = df_other_city.iloc[0:first_n_ranks, :]

                rbo_score = RBO.score(current.feature_name.tolist(), df_other_city.feature_name.tolist())
                rbo_scores_current.update({other_cities : rbo_score})

                if rbo_score > 1:
                    print('Error, an rbo score > 1 was computed!')

            df_rbo_similarity = pd.concat([df_rbo_similarity, pd.DataFrame(rbo_scores_current, index = [current_city])], sort = False)

        # reorder df_rbo_similarity to group clusters having similar rbo scores (for heatmap visualization)
        # compute the indexes/columns permutation allowing to cluster the similarities
        # but not show with clustermap because heatmap easier to control
        cm = sns.clustermap(df_rbo_similarity, cbar_kws={"label": "RBO score"})
        plt.close()
        col_names = df_rbo_similarity.columns.tolist()
        row_names = df_rbo_similarity.index.tolist()
        order_row = cm.dendrogram_row.reordered_ind
        order_col = cm.dendrogram_col.reordered_ind
        order_row_labels = []
        order_col_labels = []
        for i, j in zip(order_row, order_col):
            order_row_labels = order_row_labels + [row_names[i]]
            order_col_labels = order_col_labels + [col_names[i]]

        df_rbo_similarity = df_rbo_similarity.reindex(order_row_labels, axis = 0)
        df_rbo_similarity = df_rbo_similarity.reindex(order_col_labels, axis = 'columns')
        
        df_rbo_similarity_list = df_rbo_similarity_list + [df_rbo_similarity]
    
    return df_rbo_similarity_list, success_metrics_list



def get_merged_df_impotances(list_of_importance_df_files, files_location):
    all_importance_df = pd.DataFrame()
    idx_city = 0
    for imp_file in list_of_importance_df_files:
        df = pd.read_csv(files_location + '/' + imp_file).drop('Unnamed: 0', axis = 1)
        df['city'] = [imp_file.split('_')[0]]*df.shape[0]
        if 'feature' in df.columns.tolist():
            df['feature_name'] = df.feature.apply(lambda x: 'neigborhood' if 'neigh' in x else x.replace('_', ' '))
        all_importance_df = pd.concat([all_importance_df, df])
        idx_city += 1
    return all_importance_df

def create_save_interactive_heatmap(df_heatmap, fig_saving_folder, save_tag, showFigure = False):
    fig = go.Figure(data = go.Heatmap(
                       z = df_heatmap.values,
                       x = df_heatmap.columns.tolist(),
                       y = df_heatmap.index.tolist(),
                       hoverongaps = False, colorscale = sns.color_palette('rocket', 20).as_hex()))
    fig.update_xaxes(tickangle = -45)
    fig.update_yaxes(autorange = 'reversed')
    fig.update_layout(autosize = False, width = 800, height = 800)
    # save html 
    html_plot = pyplot(fig, filename = fig_saving_folder + '/' + save_tag + '.html', auto_open = False)

    if showFigure:
        fig.show()

def plot_heat_map(df_heatmap, fig_saving_folder, fig_title, save_tag, use_diagonal_mask = True):
    if use_diagonal_mask:
        mask = np.zeros_like(df_heatmap, dtype = np.bool)
        mask[np.triu_indices_from(mask)] = True
    
    fig = plt.gcf()
    fig.set_size_inches(8, 7)
    if use_diagonal_mask:
        chart = sns.heatmap(df_heatmap, mask = mask, square=True, cbar_kws={"shrink": .5})
    else:
        chart = sns.heatmap(df_heatmap, square=True, cbar_kws={"shrink": .5})
    chart.set_xticklabels(
                chart.get_xticklabels(), 
                rotation=45, 
                horizontalalignment='right')
    plt.title(fig_title, fontsize = 16, \
                      fontweight='bold')
    plt.savefig(fig_saving_folder + '/' + save_tag + '.pdf', bbox_inches='tight')    
    plt.show()

def matrix_treshold_binarization_preview(df_matrix, pixels_treshold):
    binary_df = df_matrix.applymap(lambda x: 0 if x < pixels_treshold else 1)
    sns.heatmap(binary_df)
    plt.title('Binarized Matrix Preview', fontsize = 14, \
                      fontweight='bold')
    plt.show()
    
def extract_clusters_from_diag_blocks(df_matrix, block_pixels_treshold, min_block_elements):
    '''
    inspired from: https://stackoverflow.com/questions/46737409/finding-connected-components-in-a-pixel-array
    '''
    binary_df = df_matrix.applymap(lambda x: 0 if x < block_pixels_treshold else 1)
    # define the 4-connected structure filter 
    connections_filter = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    # label the 4-connected regions using scipy label function (applying morphological filter)
    labeled, ncomponents = label(binary_df.values, connections_filter)
    # extract the indices of the blocks
    indices = np.indices(binary_df.values.shape).T[:,:,[1, 0]]
    
    clustered_elements = {}
    for i in range(1, ncomponents+1):
        # get indices of block (cluster) labeled with pixels values = i
        indices_block_i = indices[labeled == i]
        # if more than min_block_elements elements in block, get the elements in the block
        if len(indices_block_i) >= min_block_elements:
            unique_elements = np.unique(indices_block_i)
            clustered_elements.update({'Cluster ' + str(i) : df_matrix.columns[unique_elements].tolist()}) 
    
    return clustered_elements


def create_save_radial_plot(clustered_cities, importance_files_names, importance_files_identifier, 
        files_location, fig_saving_folder, first_n_ranks, save_tag, metric = 'multi', showFigure = False):
    df_clusters = pd.DataFrame()
    for cluster in clustered_cities.items():
        # since features importances in the clusters are very similar, take the ranking for the first city in the cluster
        cluster_cities = cluster[1]
        city           = cluster_cities[0]
        file_city_imp  = [x for x in importance_files_names if (city in x and importance_files_identifier in x)]
        if len(file_city_imp) > 1:
            print('Error!')
        tmp_imp_df = pd.read_csv(files_location + '/' + file_city_imp[0])
        if metric is not 'multi':
            tmp_imp_df = tmp_imp_df.loc[tmp_imp_df.metric == metric]

        tmp_imp_df = tmp_imp_df.iloc[0:first_n_ranks, :]
        tmp_imp_df['ranking']      = np.arange(1, first_n_ranks+1)
        tmp_imp_df['cluster']      = cluster[0]
        tmp_imp_df['feature_name'] = tmp_imp_df.feature.apply(lambda x: 'neigborhood' if 'neigh' in x else x.replace('_', ' '))
        df_clusters = pd.concat([df_clusters, tmp_imp_df])

    colors = sns.color_palette('Set1', len(clustered_cities)).as_hex()
    range_radius = [first_n_ranks, 1]
    # fig = px.line_polar(df_clusters, r = 'ranking', theta='feature_name', color = 'cluster', line_close = True,
    #                    color_discrete_sequence = colors,
    #                    range_r = range_radius)
    fig = px.scatter_polar(df_clusters, r = 'ranking', theta = 'feature_name', color = 'cluster',
                        size = 'ranking', color_discrete_sequence = colors,
                        range_r = range_radius)
    # save html
    html_plot = pyplot(fig, filename = fig_saving_folder + '/' + save_tag + '.html', auto_open = False)

    if showFigure:
        fig.show()

    
def create_save_bubble_plot(clustered_cities, importance_files_names, importance_files_identifier, 
        files_location, fig_saving_folder, first_n_ranks, save_tag, metric = 'multi', showFigure = False):
    df_clusters = pd.DataFrame()
    for cluster in clustered_cities.items():
        # since features importances in the clusters are very similar, take the ranking for the first city in the cluster
        cluster_cities = cluster[1]
        city           = cluster_cities[0]
        file_city_imp  = [x for x in importance_files_names if (city in x and importance_files_identifier in x)]
        if len(file_city_imp) > 1:
            print('Error!')
        tmp_imp_df = pd.read_csv(files_location + '/' + file_city_imp[0])
        if metric is not 'multi':
            tmp_imp_df = tmp_imp_df.loc[tmp_imp_df.metric == metric]

        tmp_imp_df = tmp_imp_df.iloc[0:first_n_ranks, :]
        tmp_imp_df['ranking']      = np.arange(1, first_n_ranks+1)
        tmp_imp_df['ranking_inv']  = np.arange(1, first_n_ranks+1)[::-1]
        tmp_imp_df['ranking_str']  = [str(x) for x in np.arange(1, first_n_ranks+1)]
        tmp_imp_df['cluster']      = cluster[0]
        tmp_imp_df['feature_name'] = tmp_imp_df.feature.apply(lambda x: 'neigborhood' if 'neigh' in x else x.replace('_', ' '))
        df_clusters = pd.concat([df_clusters, tmp_imp_df])

    colors = sns.color_palette('hls', first_n_ranks).as_hex()
    range_radius = [first_n_ranks, 1]
    #display(df_clusters)
    cat_order = [str(x) for x in np.arange(1, first_n_ranks+1)]
    
    fig = px.scatter(df_clusters, x = 'cluster', y = 'feature_name',
	                    size = 'ranking_inv', color = 'ranking_str',
                        hover_name = 'feature_name', size_max=35, 
                        labels = {'feature_name':'feature name', 'ranking_str':'rank', 'ranking_inv':'bubble size'},
                        width = 800, category_orders = {'ranking_str':cat_order},
                        color_discrete_sequence = colors)
    '''
    fig = px.scatter_polar(df_clusters, r = 'ranking', theta='feature_name', color = 'cluster',
                        size = 'ranking', color_discrete_sequence = colors,
                        range_r = range_radius)
    '''
    # save html
    html_plot = pyplot(fig, filename = fig_saving_folder + '/' + save_tag + '.html', auto_open = False)

    if showFigure:
        fig.show()