import os
import time
import pydot
import folium
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import preprocessing
import plotly.graph_objects as go
from folium.plugins import MiniMap
from IPython.core.display import display
from plotly.offline import plot as pyplot
from sklearn.tree import export_graphviz
from sklearn.utils.validation import check_array
from sklearn.ensemble import RandomForestRegressor
from treeinterpreter import treeinterpreter as ti, utils
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, make_scorer, mean_squared_error, mean_absolute_error, median_absolute_error, r2_score


class FeaturesTools():
    """
    FeatureTools is a class defining tools allowing to run ML analysis on the features contained in a data set and to generate some graphic outputs

    Attributes:
    -----------
    df : pandas DataFrame
        the input data
    random_seed : int
    train_fraction : float
        the fraction of the input data that is used for training
    label_name : string
        the name of the input data column that is used as label in ML algorithms
    train_set : pandas DataFrame
        only present if df is specified
    test_set : pandas DataFrame
        only present if df is specified
    train_label : pandas Series
        only present if df is specified
    test_label : pandas Series
        only present if df is specified
    
    
    Methods:
    -----------
    split_df(df, label_name, train_fraction, random_seed)
        splits an input data set into train and test sets and labels (targets)
        
    normalize_features(df = None)
        normalizes an input data set along features axis (normalize each feature indipendently)
        
    standardize_features(df = None)
        standardizes an input data set along features axis (standardize each feature indipendently)
        
    importance_df(column_names, importances, std)
        assembles the method parameters nto a dataframe indicating the importance of each feature
        
    randomForestAnalysis(train_data = None, train_labels = None, test_data = None, test_labels = None, 
                             seed = None, n_trees = 500, plotResults = False, plotTree = False, tuneModelParameters = False)
        allows to perform a complete machine learning analysis using random forests. 
        
    treeInterpreter(rf_model, df)
        allows the interpretation of a tree and to determine the contribution of each feature (node) to the final predictions
        
    correlationAnalysis(df = None, method = 'pearson', plotMatrix = True, printMaxCorr = True)
        performs correlation analysis of the features (columns) of the input data frame
        
    word_cloud(mask_file_path, df_importance, out_image_name)
        generates a word cloud representation of an input data frame

    interactive_sankey(df, figureTitle)
        generates interactive sankey diagrams of the input data frame
          
    """
    
    def __init__(self, df = None, label_name = None, train_fraction = 0.7, random_seed = np.random.randint(low = 0, high = np.iinfo(np.int32).max)):
        """class constructor
        If a data frame is specified as input, the split into train and test sets and labels will be automatically performed and stored as class attributes.
        
        Parameters:
        -----------
        df : pandas DataFrame
            Default: None (empty class)
        label_name : string
            name of the column in df that will be used as label for machine learning, if df is specified. Default: None
        train_fraction : float
            fraction of df used as training set. Default: 0.7
        random_seed : int
            the used random seed. Default: an automatically generated integer random number between 0 and np.iinfo(np.int32).max

        Returns:
        -----------
        -

        """
        self.df             = df
        self.random_seed    = random_seed
        self.train_fraction = train_fraction
        if isinstance(label_name, str):
            self.label_name = [label_name]
        else:
            self.label_name = label_name
        if (df is not None) and (label_name is not None):
            for key, value in self.split_df():
            #for key, value in self.split_df(self.df, self.label_name, self.train_fraction, self.random_seed):
                setattr(self, key, value)
    
    
    def split_df(self, df = None, label_name = None, train_fraction = None, random_seed = None):
        """split the input data frame into a train set, test set, train label, test label 
        
        Parameters:
        -----------
        df : pandas DataFrame
            the input data. If not specified the corresponding class attribute is used.
        label_name : string
            the name of the column that is used as label. If not specified the corresponding class attribute is used.
        train_fraction : float
            the fraction of the data used as train data. If not specified the corresponding class attribute is used.
        random_seed : int
            the used tandom seed. Default : None (the class attribute is used). If not specified the corresponding class attribute is used.

        Returns:
        -----------
        [train_label, train_set, test_label, test_set] : pandas objects
             the split objects

        """
        
        if random_seed is None:
            #np.random.seed(self.random_seed)
            random_seed  = self.random_seed
        #else:
        #    np.random.seed(random_seed)
        
        if train_fraction is None:
            train_fraction = self.train_fraction
        
        if label_name is None:
            label_name = self.label_name
        
        if df is None:
            df = self.df

        
        #msk = np.random.rand(df.shape[0]) < train_fraction
        
        x_train, x_test, y_train, y_test = train_test_split(df[df.columns.difference(label_name)], df[label_name], train_size = train_fraction,
            test_size = 1 - train_fraction, random_state = random_seed)

        train_set   = ['train_set',   x_train]
        train_label = ['train_label', y_train]
        test_set    = ['test_set',    x_test]
        test_label  = ['test_label',  y_test]

        return [train_label, train_set, test_label, test_set]
    
    
    def preprocess_features(self, preprocessing_type, df = None):
        """preprocess features according to the method specified by preprocessing_type.


        Parameters:
        -----------
        df : pandas DataFrame
        preprocessing_type : string
            preprocessing_type = normalize          -> scaling features of the input data frame individually to have unit norm
            preprocessing_type = standardize        -> standartizes features of the input data frame individually to have zero mean, unit variance
            preprocessing_type = scale_unit         -> scaling features of the input data frame individually, in the [0, 1] range
            preprocessing_type = standardize_robust -> scaling features of the input data frame individually, robust to outliers!

        Returns:
        -----------
        norm: pandas DataFrame
            the normalized data frame

        """
        
        print('> Running normalize_features...\n')
        if df is None:
            df = self.df
        
        if preprocessing_type == 'normalize':
            transf = pd.DataFrame(preprocessing.normalize(df, axis = 0)) 
            transf.columns = df.columns
        elif preprocessing_type == 'standardize':
            scaler = preprocessing.StandardScaler()
            transf = pd.DataFrame(scaler.fit_transform(df.values))
            transf.columns = df.columns 
        elif preprocessing_type == 'scale_unit':
            scaler = preprocessing.MinMaxScaler()
            transf = pd.DataFrame(scaler.fit_transform(df.values))
            transf.columns = df.columns
        elif preprocessing_type == 'standardize_robust':
            scaler = preprocessing.RobustScaler()
            transf = pd.DataFrame(scaler.fit_transform(df.values))
            transf.columns = df.columns

        return transf
    
    
    def importance_df(self, column_names, importances, std):
        """aggregate the input parameters into a data frame indicating the importance of each feature and the std
        
        Parameters:
        -----------
        column_names : array of string
            names of the features
        importances: array of float
            array containing the importance of each feature
        std: array of float
            array containing the std of each feature

        Returns:
        -----------
        df_importance : pandas DataFrame

        """
        
        print('> Running importance_df...\n')
        df_importance = pd.DataFrame({'feature': column_names,
                           'feature_importance': importances,
                           'std' : std}) \
               .sort_values('feature_importance', ascending = False) \
               .reset_index(drop = True)
        return df_importance
    
    
    
    def randomForestAnalysis(self, train_data = None, train_labels = None, test_data = None, test_labels = None, 
                             seed = None, n_trees = 500, plotResults = [False, False, False, False], tuneModelParameters = False):
        
        """performs a complete random forest analysis
        
        1) build a random forest model. If tuneModelParameters = True, hyperparameters optimization is performed and the best model is selected.
        2) evaluate the model, extract features importance, ...
        3) plot results
        
        
        Parameters:
        -----------
        train_data : pandas DataFrame
            if not specified the class attribute is used
        train_labels : pandas Series
            if not specified the class attribute is used
        test_data : pandas DataFrame
            if not specified the class attribute is used
        test_labels : pandas Series
            if not specified the class attribute is used
        seed : int
            if not specified the class attribute is used
        n_trees : int
            number of trees used if tuneModelparameters is False. default: 500
        plotResults : array of bool
            plot some of the analysis results
            plotResults = [PlotFeaturesImportance, PlotCumulativeImportance, PlotPredictionsVsLabels, PlotTree]
        tuneModelparameters : bool
            if True, an optimization of the sklearn random forest model hyperparameters is performed and the best model is extracted.
            A grid search is performed
        

        Returns:
        -----------
        df_importance : pandas DataFrame
            a data frame indicating the importance of each feature
        rf : sklearn random forest model
            the built sklearn random forest model
        """
        
        print('> Running randomForestAnalysis...\n')
        
        if train_data is None:
            train_data = self.train_set
        if train_labels is None:
            train_labels = self.train_label
        if test_data is None:
            test_data = self.test_set
        if test_labels is None:
            test_labels = self.test_label
        if seed is None:
            seed = self.random_seed
        
        # advantage of the random forest:
        # The random forest performs implicit feature selection because it splits nodes on the most important variables.
        # The random forest feature importances can be used to reduce the number of variables in the problem.
        # In addition to potentially increasing performance, reducing the number of features will shorten the run time of the model.
        #
        # In decision trees, every node is a condition of how to split values in a single feature, so that similar values of the 
        # dependent variable end up in the same set after the split. The condition is based on impurity, which in case of classification
        # problems is Gini impurity/information gain (entropy), while for regression trees its variance. So when training a tree we can 
        # compute how much each feature contributes to decreasing the weighted impurity.
        # feature_importances_ in Scikit-Learn is based on that logic, but in the case of Random Forest, we are talking about averaging
        # the decrease in impurity over trees.
        # > Cons: biased approach, as it has a tendency to inflate the importance of continuous features or high-cardinality categorical variables
        # 
        # Other more complex dimensionality reductions methods (do a good job of decreasing the number of features while not decreasing 
        # information, but they transform the features such that they no longer represent our measured variables):
        # > PCA
        # > ICA
        
        # tune the model hyperparameters to run the best model. 
        # Use Random Search Training: not every combination is tested, but some are selected at random to sample a wide range of values.
        # Otherwise: too long!
        if tuneModelParameters:
            t0 = time.time()
            
            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2100, num = 21)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(1, 111, num = 20)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            # bootstrap = [True] # can onla use true, otherwise oob_score can not be computed and we use this score to evaluate the model
            # When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble
            # (similar to boosted trees), otherwise, just fit a whole new forest.
            warm_start = [False, True]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf, 
                           'warm_start': warm_start}
                           #'bootstrap': bootstrap}
                        
            # Random grid to search for best hyperparameters
            # First create the base model to tune
            rf_ = RandomForestRegressor(oob_score = True)

            # Random search of parameters, using 4 fold cross validation
            #rf_random = RandomizedSearchCV(estimator = rf_, param_distributions = random_grid, n_iter = 2, cv = 2, verbose=2, random_state = seed, 
            #                n_jobs = -1, scoring = {'MSE': make_scorer(mean_squared_error)}, refit = 'MSE', return_train_score = True)
            rf_random = RandomizedSearchCV(estimator = rf_, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, random_state = seed, 
                            n_jobs = -1, scoring = {'MSE': make_scorer(mean_squared_error)}, refit = 'MSE', return_train_score = True)
        
            # Fit the random search model
            if train_labels.shape[1] == 1:
                rf_random.fit(train_data, train_labels.values.ravel())
            else:
                rf_random.fit(train_data, train_labels)
            
            # print best parameters:
            print('\nThe best RF model hyperparameters from RandomizedSearchCV are:')
            print(rf_random.best_params_)
            print('\n')
            
            # select one tree for visualization
            visual_tree = rf_random.best_estimator_[1]
            
            rf = rf_random.best_estimator_
            #rf = rf.fit(train_data, train_labels)
        

            elapsed_time = time.time() - t0
            print('Elapsed time ... : ' + str(elapsed_time))
        
        else:
            # > n_estimators = number of trees in the forest
            # > The criterion is by default mse
            # > n_jobs = -1 to use all the processors if multicore cpu
            # > set random_state (random seed) to 1 for reproducibility
            # > OOB is the out-of-bag score, computed on the leaved-out samples when using bootstrapping. If true, use out-of-bag samples 
            #   to estimate the R^2 on unseen data.
            rf = RandomForestRegressor(n_estimators = 100,
                                       n_jobs = -1,
                                       oob_score = True,
                                       bootstrap = True,
                                       random_state = seed)
            
            if train_labels.shape[1] == 1:
                rf.fit(train_data, train_labels.values.ravel())
            else:
                rf.fit(train_data, train_labels)

            
            # select one tree for visualization
            visual_tree = rf.estimators_[1]
        
        # predictions on test data:
        predictions = rf.predict(test_data)

        # useful post about errors: 
        # https://stats.stackexchange.com/questions/327464/mape-vs-r-squared-in-regression-models

        df_err = pd.DataFrame()

        col_names = test_labels.columns
        col_names = np.append(col_names, 'AVG')

        # R^2 error
        r2_err = r2_score(test_labels, predictions, multioutput = 'raw_values')
        df_err = df_err.append(pd.Series(np.append(r2_err, np.mean(r2_err)), name = 'R^2 error'))

        # MSE error:
        mse = mean_squared_error(test_labels, predictions, multioutput = 'raw_values')
        df_err = df_err.append(pd.Series(np.append(mse, np.mean(mse)), name = 'MSE error'))

        # MAE error:
        mae = mean_absolute_error(test_labels, predictions, multioutput = 'raw_values')
        df_err = df_err.append(pd.Series(np.append(mae, np.mean(mae)), name = 'MAE error'))
        
        # MAPE error:
        y_true, y_pred = np.array(test_labels), np.array(predictions)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        df_err = df_err.append(pd.Series(np.append(mape, np.mean(mape)), name = 'MAPE error'))

        # median absolute error (does not support multioutput on scikit-learn version is 0.20.1.):
        med_err = []
        if test_labels.shape[1]>1:
            for i in range(test_labels.shape[1]):
                med_err = np.append(med_err, median_absolute_error(test_labels.iloc[:, i], predictions[:, i]))
        else:
            med_err = np.append(med_err, median_absolute_error(test_labels, predictions))

        df_err = df_err.append(pd.Series(np.append(med_err, np.mean(med_err)), name = 'Median Absolute error'))

        df_err.columns = col_names
        
        # mean absolute percentage error (MAPE)
        # mape = np.mean(100 * (errors / test_labels))
        
        
        # build df with features importance
        importance = rf.feature_importances_
        std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
        df_importance = self.importance_df(train_data.columns, importance, std)
        
        # cumulative importance
        cumulative_importance = df_importance['feature_importance'].cumsum()
        
        # No overfit is observed if the OOB score and the Test (Validation) score are very similar
        
        print('> randomForestAnalysis results...')
        # print('R^2 Training Score:     %f' %(rf.score(train_data, train_labels)))
        print('R^2 OOB Score:          %f' %(rf.oob_score_))
        # print('R^2 Test Score:         %f' %(rf.score(test_data, test_labels)))
        #print('MSE Error:              %f' %(mse))
        #print('MAE Error:              %f' %(mae))
        #print('Median Absolute Error:  %f (insensible to outliers)' %(med_err))
        # print('Accuracy:               %f%%\n\n' %(round(100 - mape, 2)))
        #print("Feature ranking:")
        #display(df_importance)
        print('\n> Performances comparison for the target(s) variables:\n')
        print(tabulate(df_err, headers='keys', tablefmt='psql', numalign = 'center', stralign = 'center'))
        
        if plotResults[0]:
            # error bars are so small that are not visible
            sns.barplot(x = 'feature_importance', y = 'feature', color='skyblue', data = df_importance, orient = 'h', ci = 'sd', capsize=.2)
            fig = plt.gcf()
            fig.set_size_inches(20, 15)
            plt.xlabel('Feature Importance', size = 22)
            plt.ylabel('Features', size = 22)
            plt.title('Feature importance of each feature', size = 22)
            plt.tick_params(labelsize = 10)
            plt.show()

        if plotResults[1]:
            # plot Cumulative importance.
            x_values = np.arange(0, len(cumulative_importance))
            plt.plot(x_values, cumulative_importance, 'k-')
            fig = plt.gcf()
            fig.set_size_inches(20, 6)
            # Draw line at 95% of importance retained
            l1 = plt.hlines(y = 0.95, xmin = 0, xmax = len(cumulative_importance), color = 'r', linestyles = 'dashed')
            ax = plt.gca()
            ax.set_xticks(x_values)
            ax.set_xticklabels(df_importance.feature, rotation = 45, ha = 'right')
            plt.xlabel('Features', size = 22)
            plt.ylabel('Cumulative Importance', size = 22)
            plt.title('Cumulative importantce of the features', size = 22)
            plt.tick_params(labelsize = 10)
            plt.legend([l1], ['95% threshold'], loc = 'center right', fontsize = 20)
            plt.show()

        if plotResults[2]:
            # plot labels vs preictions
            fig = plt.figure(figsize = (20, 6))
            x_values = np.arange(0, len(predictions))
            for ff in range(test_labels.shape[1]):
                plt.subplot(1, test_labels.shape[1], ff + 1)
                plt.scatter(x_values, test_labels.iloc[:, ff], label = 'target')
                if test_labels.shape[1]>1:
                    plt.scatter(x_values, predictions[:, ff], label = 'prediction')
                else:
                    plt.scatter(x_values, predictions, label = 'prediction')
                fig = plt.gcf()
                fig.set_size_inches(20, 6)
                plt.ylabel(test_labels.columns[ff], size = 22)
                plt.xlabel('Sample ID', size = 22)
                plt.legend(fontsize = 20)
                plt.tick_params(labelsize = 10)

        if plotResults[3]:
            # visualize the tree
            export_graphviz(visual_tree, out_file = './RandomTree.dot', feature_names = train_data.columns, 
                precision = 2, filled = True, rounded = True, max_depth = None)
            (graph, ) = pydot.graph_from_dot_file('./RandomTree.dot')
            graph.write_png('./RandomTree.png')
        

            # Limit depth of tree to 3 levels
            # rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
            # rf_small.fit(train_features, train_labels)
            # Extract the small tree
            # tree_small = rf_small.estimators_[5]
            # Save the tree as a png image
            # export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
            # (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
            # graph.write_png('small_tree.png');
    
        if tuneModelParameters:
            return df_importance, df_err, rf, rf_random.cv_results_, rf_random.best_params_
        else:
            return df_importance, df_err, rf
        
        
    def treeInterpreter(self, rf_model, df = None):
        """allows the interpretation of a sklearn random forest model
        
        Parameters:
        -----------
        rf_model : sklearn random forest model
        df : pandas DataFrame

        Returns:
        -----------
        prediction : array
            see details in the code commentary
        bias : array
            see details in the code commentary
        contributions : array
            see details in the code commentary

        """
        
        print('> Running treeInterpreter...\n')

        if rf_model.n_outputs_ > 1:
            print('SORRY, tree interpretation not possible for a multilabel model using treeInterpreter method from FeaturesTools class !!!\n')
            return [], [], []
        
        # INTERPRETING RANDOM FORESTS:
        # https://blog.datadive.net/random-forest-interpretation-with-scikit-learn/
        # http://blog.datadive.net/interpreting-random-forests/
        #
        # > Inspecting the rendom foprest model using treeinterpreter allow to extract the prediction paths for each individual prediction 
        # and decompose the predictions via inspecting the paths. Allow to to know why the model made a decision, i.e. how much each feature 
        # contributed to the final outcome.
        #
        # > The predictions can be decomposed into the bias term (which is just the trainset mean) and individual feature contributions, 
        # so we see which features contributed to the difference and by how much.
        #
        # > To check if the decomposition is correct: bias and contributions need to sum up to the predicted value: prediction must be = 
        # biases + np.sum(contributions, axis=1) where bias = trainset mean
        #
        # > Understanding the features contribution: The regression tree is composed of nodes, each with some specific rule. The prediction
        # is computed for a followed path composed of specific nodes as:
        # Prediction: Y ≈ X0 (train set mean = bias, is the value at the first upstream node; if we stop here we get only the mean; need to 
        # continue to get precise prediction!) + X1(gain from node1) + X2(gain from node2) - X3(loss from node3) + ... - ...
        # X1, ..., XN are the features contributions
        # (see picture at http://blog.datadive.net/interpreting-random-forests/)
        #
        # > This approach can be very useful is when comparing two datasets. For example:
        # Understanding the exact reasons why estimated values are different on two datasets, for example what contributes to estimated 
        # house prices being different in two neighborhoods.
        # prediction1, bias1, contributions1 = ti.predict(rf, ds1)
        # prediction2, bias2, contributions2 = ti.predict(rf, ds2)
        # use the underlying trees in Random Forest to explain how each feature contributes to the end value. 
        
        if df is None:
            df = self.df
        
        prediction, bias, contributions = ti.predict(rf_model, df.values)
        
        return prediction, bias, contributions
        
        
      
    def correlationAnalysis(self, df = None, method = 'pearson', plotMatrix = True, printMaxCorr = True):
        """computes the correlation matrix of the input data
        
        Parameters:
        -----------
        df : pandas DataFrame
            if None the class attribute is used
        method : string
            method used to compute correlation. Can be for example perason or sparman
        plotMatrix : bool
            if true, plot the lower half diagonal of the correlation matrix with an heatmap
        printMaxCorr : bool

        Returns:
        -----------
        corr : matrix

        """
        
        print('> Running correlationAnalysis...\n')
        
        if df is None:
            df = self.df
            
        corr = df.corr(method = 'pearson')

        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        
        if plotMatrix:
            #chart = sns.heatmap(corr, mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=True, yticklabels=True)
            chart = sns.heatmap(abs(corr), mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=True, yticklabels=True)
            fig = plt.gcf()
            fig.set_size_inches(20, 20)
            plt.tick_params(labelsize=10)
            chart.set_xticklabels(
                chart.get_xticklabels(), 
                rotation=45, 
                horizontalalignment='right')
            plt.title('Features correlation matrix - ' + method, fontsize = 22, \
                      fontweight='bold')
            plt.show()
        
        # print: variable... is max correlated with....
        #if printMaxCorr:
            #pass

        return corr
    
    
    def word_cloud(self, mask_file_path, df_importance, out_image_name):
        """generates a word cloud representation of the input data
        
        Parameters:
        -----------
        mask_file_path : string
            complete path to the file used as shape-mask for the word cloud. !! The image must be binary, 8-bits. 
            If not the case, convert it before for example with imagej !!
        df_importance : pandas DataFrame
            input data visualized in the word cloud representation. From method importance_df(...)
        out_image_name : string
            complete path of the output file name, saved as .png

        Returns:
        -----------
        -
        """
        
        print('> Running word_cloud...\n')
        mask = np.array(Image.open(mask_file_path))
        
        text = [w.replace('_', ' ') for w in df_importance.feature.tolist()]
        wext_weight = {}
        for key, weight in zip(text, df_importance.feature_importance.tolist()):
            wext_weight[key] = weight
        
        wc = WordCloud(background_color = 'black', max_words = 1000, mask = mask, contour_color = '#FB404D',
                      contour_width = 2).generate_from_frequencies(wext_weight)
        #wc.to_file('./' + out_image_name + '.png')
        plt.figure(figsize=[20,10])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.title('Word cloud for features names where size reflects the feature importance', fontsize = 16, 
                  fontweight='bold')
        plt.savefig('./' + out_image_name + '.png')
        plt.show()
        
    
    def interactive_sankey(self, df_contributions, figureTitle):
        """builds an interative sankey diagram from the input data
        
        Parameters:
        -----------
        df : pandas DataFrame
            df must contain columns: ['feature_source', 'feature_target', 'contribution', 'source', 'target', labels]
        figureTitle : string
            title of the displayed figure

        Returns:
        -----------
        -
        """
        
        print('> Running interactive_sankey...\n')
        # https://plot.ly/~alishobeiri/1591/plotly-sankey-diagrams/#/
        
        data_trace = dict(
            type='sankey',
            domain = dict(
              x =  [0,1],
              y =  [0,1]
            ),
            orientation = "h",
            valueformat = ".0f",
            node = dict(
              pad = 10,
              thickness = 30,
              line = dict(
                color = "black",
                width = 0
              ),
              label =  df_contributions.labels,
              color = df_contributions.color.dropna(axis=0)
            ),
            link = dict(
              source = df_contributions.source,
              target = df_contributions.target,
              value  = df_contributions.contribution,
              color = df_contributions.linkColor,
          )
        )

        layout =  dict(
            title = figureTitle,
            height = 1000,
            font = dict(
              size = 10
            ),    
        )

        fig = go.Figure(data=[data_trace], layout=layout)
        
        html_sankey = pyplot(fig, filename='sankey_plot.html', auto_open=False)
        with open('sankey_plot.html', 'r') as f:
            html = f.read()
        fig.show()
    
    def map_to_color(self, color_palette, list_of_values, min_range = None, max_range = None):
        """map a list of values to discrete colors
        If the min_range is specified, any value smaller than this value will be assigned to the minimal color. Similarly for max_range.
        
        Parameters:
        -----------
        color_palette  : array of hex-encoded colors
        list_of_values : array
            the list of values that have to be mapped to a color
        min_range      : float
            the minimal value associated to the minimum color (default = None)
        max_range      : float
            the maximal value associated to the minimum color (default = None)
        
        Returns:
        -----------
        mapped_colors : array
            the list of colors mapped to the input values
        """
        
        if min_range == None:
            min_range = np.min(list_of_values)
        if max_range == None:
            max_range = np.max(list_of_values)
        
        intervals = np.linspace(min_range, max_range, len(color_palette)+1)

        mapped_colors = []
        for i in list_of_values:
            # we have N-1 colors
            for c in range (0, len(intervals) - 1):
                if c == 0 and i < intervals[c]:
                    mapped_colors = np.append(mapped_colors, color_palette[c])
                if c < len(intervals) and i >= intervals[c] and i < intervals[c + 1]:
                    mapped_colors = np.append(mapped_colors, color_palette[c])
                    break
                elif c == len(intervals) - 2 and i >= intervals[c]:
                    mapped_colors = np.append(mapped_colors, color_palette[c])
        return mapped_colors
    


        