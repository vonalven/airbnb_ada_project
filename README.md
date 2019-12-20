# What makes an Airbnb successful?

LINK TO THE DATASTORY: https://vonalven.github.io/

# Run the projet

Run the [main.ipynb](https://github.com/vonalven/airbnb_ada_project/blob/master/main.ipynb) jupyter notebook 
    
*For visualization purposes, this notebook should be opened with JupyterLab*


# Abstract
Airbnb is a world-wide known online lodging marketplace which is well established in most of the principal touristic destinations. Since its foundation in 2008, the company experienced a continuous growth with a boom after 2012, ending up to a revenue of 2.6 billions US dollars in 2017, 12736 employees in 2019 and an astonishing 2 millions people lodging with Airbnb each night in October of this year. In the latest years Airbnb has often been targeted by some legal process and was put at the center of local scandals related to over-presence causing discriminations in the renting real estates marketplaces. 

In this project, we would like to determine what makes an Airbnb housing successful. The definition of success can be hard to define and can have several interpretations. According to the Cambridge dictionary, success is defined as "something that achieves positive results". For our analysis, we will put ourselves in the shoes of both the guests and the hosts. For hosts, one could imagine that this translates in a good evaluation of the proposed services or a frequently booked listing. On the other hand, customers demonstrate their appreciation through a very positive review. 

Thus, we will seek to find metrics within our data that give us: 

* The score of the listing.
* The sentiment of the reviews.
* The occupancy rate of the listing.

The general idea here is to provide an interesting analysis and to expose it with impacting graphical representations. The main goal is to provide a simple and user-friendly overview of the extracted results. 

# Research questions

The underlying questions of our main problem, what defines a successful Airbnb, are:
* What do people seek for when booking an airbnb: what variables have the largest impact on a listing ?
* How are successful listings distributed in the cities of interest ?
* What can be learned from the "big players" and "international players" of the Airbnb platform ? re the same parameters important for all cities ?

# Datasets

The Airbnb datasets we will use are accessible and downloadable [here](http://insideairbnb.com/index.html). Inside Airbnb is an independent non-commercial website that groups Airbnb insides and allows to explore via some virtual tools how Airbnb is used in 101 cities around the world. 
For each city the main collected informations are:
* detailed listing data
(host information, type of lodgement, type of beds, provided services, price and fees, ...)
* detailed calendar data
* detailed calendar data
* detailed review data
* summary information and metrics for listings, which is good for visualisations
* summary review data and listing ID
* neighbourhood list for geo filter, sourced from city or open source GIS files
* GeoJSON file of neighbourhoods of the city

In total a list of 106 features is available, without accounting for geographic information. 

Information about how the data was collected and pre-processed is available in the source-website. The datasets are highly-comprehensive, very complete and usable.

To provide additional insight on studied cities, our dataset will be completed by public transportation datasets. The ones used in this project are accessible and downloadable [here](https://www.citylines.co/data). This open-source platform is constantly updated and all the data cover several years up to the present days. 
For each city, the main collected information consists of:
* transportation systems
* transportation lines
* transportation sections
* transportation section lines
* transportation stations
* transportation modes

All the data is downloaded as .csv files and handled using pandas.


# Organisation of the repository

```
|
|   README.md                                                               > README of the project  
|   main.ipynb                                                              > performs the entire analysis, starting from data exploration to single city study
|   
+---img
|
|   +---cities_pictures_jpg  
| 
| 	    *cityname* .jpg							                            > JPEG format pictures of cities commun to both datasets to vizualise the clusters of   |                                                                             cities
|  
|   +---html_files 
|
| 	    Airbnb_world_map.html					                            > Produced world map showing the cities from the data set in the globe.
| 	    CITYNAME_sankey_plot_threshol=0.01.html	                            > Produced sankey plot showing features importance for each success metric for the      |                                                                             city
| 	    coumpound_importance_clusters_ TYPEOFPLOT.html		                > Plots showing features importance for the compound metrics. The types of plots are    |                                                                             polar plot or bubble plot.
| 	    review_per_month_importance_clusters_TYPEOFPLOT.html	            > Plots showing features importance for the review_per_month metrics. The types of      |                                                                             plots are polar plot or bubble plot.
| 	    review_score_rating_importance_clusters_TYPEOFPLOT.html	            > Plots showing features importance for the review_score_ratings metrics. The types of  |                                                                             plots are polar plot or bubble plot.
| 	    multi_target_rating_importance_clusters_TYPEOFPLOT.html	            > Plots showing features importance for the all three metrics simultaneously            |                                                                             (multi-target analysis). The types of plots are polar plot or bubble plot.
| 	    compound_importances_ranking_similarity_interactive.html	        > Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             the compound metric.
| 	    review_per_month_importances_ranking_similarity_interactive.html 	> Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             the review_per_month metric.
| 	    review_score_rating_importances_ranking_similarity_interactive.html	> Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             the review_score_rating metric.
| 	    multi_target_rating_importances_ranking_similarity_interactive.html	> Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             all three metrics simultaneously (multi-target analysis).
| 	    review_per_month_importances_ranking_similarity_interactive.html    > Heatmap showing city clusters based on similarity in terms of features importance for |                                                                            the review_per_month metric.
| 	    review_score_rating_importances_ranking_similarity_interactive.html	> Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             the review_score_rating metric.
| 	    multi_target_rating_importances_ranking_similarity_interactive.html	> Heatmap showing city clusters based on similarity in terms of features importance for |                                                                             all three metrics simultaneously (multi-target analysis).
|
|
|   +---jpg_files                                                           > Contains all the produced results in JPEG format
|
|   +---pdf_files                                                           > PDF version of heatmaps generated during analysis to save some memory space
|
+---notebooks
|
|   GetCommentAnalysis.ipynb				                                > This notebook allows to get the complete analysis of comments data from a given       |                                                                             city's Airbnbs
|   MapClass.ipynb 					                                        > This notebook allows to easily handle and create folium maps
| 	MapMovies.ipynb						                                    > This notebook does animated folium maps
|	NLP_Metrics.ipynb					                                    > This notebook generates NLP (natural language processing) metrics: positivity,        |                                                                             negativity and compound.
|	NearestStation_simple.ipynb				                                > In this notebook, the code to calculate the distance to the nearest stations is       |                                                                             tested on the Amsterdam listings
|   Transport_Download.ipynb                                                > This notebook allows to download transports dataset
| 	airbnb_DataSet_Download.ipynb				                            > Notebook to dowload the data set from insideairbnb.com
|   comentsAnalysis.ipynb					                                > In this notebook, comments data set of Amsterdam listings is explored, cleaned. Then  |                                                                             the sentiment of each comment is computed.
| 	dist_to_station.ipynb						                            > Exploratory analysis of the public transport data set
| 	success_metrics_exploration.ipynb			                            > Exploratory analysis of the Amsterdam dataset to find out what could be used to       |                                                                             define success -> the success metrics.
+---src
|
| 	cleaning_utility.py				                                        > Class defining tools to clean the airbnb data set
| 	comment_analysis.py			                                            > Contains all the functions necessary to perform comment analysis of the airbnb reviews
| 	feature_tools.py 				                                        > Class defining tools allowing to run ML analysis on the features contained in a data  |                                                                             set and to generate some graphic outputs
| 	hidden_print.py					                                        > Class to hide the print of some code sections
| 	main_functions.py				                                        > Contains the functions called in the main for the whole analysis
| 	prepare_clean_data.py			                                        > Contains all the functions necessary to clean the datasets
| 	stations.py					                                            > Contains the function to use the public transport dataset to get their localization   |                                                                             and compute distances from stations to listing
| 	stations_distance_utilities.py		                                    > Contains functions to calculate distances from stations to listing
|   
+---RBO                                                                     > folder containing files to compute similarities for clustering + to compute the Rank  |                                                                             Biased Overlab (RB0)
|
|
```  

# List of tasks for milestone 3

* Include transportations datasets in our analysis

* Improve RF model, for example with the already implemented hyperparameter tuning or to allow multitarget predictions

* Analyse other cities, go on the international level

* An interesting visual representation would be to show some folium maps, exploiting the already coded classes ([MapClass.ipynb](https://github.com/vonalven/airbnb_ada_project/blob/master/notebooks/MapClass.ipynb) and [MapMovies.ipynb](https://github.com/vonalven/airbnb_ada_project/blob/master/notebooks/MapMovies.ipynb)) 

# Team's investment

Everyone has contributed equally to the project. Even though tasks were distributed among the team mates for efficiency, every aspect was studied by every student.

