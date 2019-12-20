# What makes an Airbnb successful?

The Data Story can be found [here](https://vonalven.github.io).

# Run the projet

Run the file [main.ipynb](https://github.com/vonalven/airbnb_ada_project/blob/master/main.ipynb). 
    
*For visualization purposes, this notebook should be opened with JupyterLab.*


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
|   main.ipynb                                                              > performs the entire analysis
|   
+---img
|
|   +---cities_pictures_jpg                                                 > JPEG format pictures of cities 			                              
|  
|   +---html_files                                                          > generated plots and heatmaps 
|
|   +---jpg_files                                                           > Contains all the produced results in JPEG format
|
|   +---pdf_files                                                           > PDF version of generated heatmaps 
|
+---notebooks
|
|   GetCommentAnalysis.ipynb                                                > allows sentiment analysis of comments
|   MapClass.ipynb                                                          > allows to easily handle and create folium maps
|   MapMovies.ipynb                                                         > allows to create animated folium maps
|   NLP_Metrics.ipynb                                                       > generates NLP metric of sentiment analysis
|   NearestStation_simple.ipynb                                             > allows to compute the distance to stations
|   Transport_Download.ipynb                                                > allows to download transports dataset
|   airbnb_DataSet_Download.ipynb                                           > allows to dowload dataset from insideairbnb.com
|   comentsAnalysis.ipyb                                                    > contains sentiment analysis of comments
|   dist_to_station.ipynb                                                   > exploration of the public transports dataset
|   success_metrics_exploration.ipynb                                       > Exploratory analysis of the Amsterdam dataset
|
+---src
|
|   cleaning_utility.py                                                     > class defining tools to clean airbnb dataset
|   comment_analysis.py                                                     > contains functionsto perform comment analysis
|   feature_tools.py                                                        > class defining tools allowing ML + related plots 
|   hidden_print.py                                                         > class to hide the print of some code sections
|   main_functions.py                                                       > contains functions for complete analysis in main
|   prepare_clean_data.py                                                   > contains functions to clean the datasets
|   stations.py                                                             > contains the function to use the public 
|   stations_distance_utilities.py                                          > contains functions to compute station distances
|   
+---RBO                                                                     > files to compute similarities for clustering +  |                                                                             to compute the Rank Biased Overlab (RB0)
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

