# What makes an Airbnb successful?

![](./img/airbnb-part-de-marche-paris.jpg)
(Image retrieved from: https://pagtour.info/on-evoque-souvent-airbnb-mais-rarement-les-problemes-des-clients/)

# Milestone 2 Remark:
For this milestone, the submitted notebook that has to be evaluated is called "main.jpynb"

# Abstract
Airbnb is a world-wide known online lodging marketplace which is well established in most of the principal touristic destinations. Since its foundation in 2008, the company experienced a continuous growth with a boom after 2012, ending up to a revenue of 2.6 billions US dollars in 2017, 12736 employees in 2019 and an astonishing 2 millions peoples lodging with Airbnb each night in October of this year. In the latest years Airbnb has often been targeted by some legal process and was put at the center of local scandals related to over-presence causing discriminations in the renting real estates marketplaces. 

In this project, we would like to determine what makes a successful Airbnb housing. The definition of success can be hard to define and can have several interpretations. According to the Cambridge dictionary, success is defined as "something that achieves positive results". For our analysis, we will put ourselves in the shoes of both the customers and the hosts. For hosts one could imagine that this translates in a good evaluation of the proposed services. On the opposite side, customers additionally could demonstrate their appreciation through a very positive review. Thus, success will be defined by three criteria: 
* The score of the listing
* The positivness of the reviews
* The occupancy rate of the listing

The general idea here is to provide an interesting analysis and to expose it with impacting graphical representations. The main goal is to provide a simple and user-friendly overview of the extracted results. 

# Research questions
The underlying questions of our main problem, what defines a successful Airbnb, are:
* What do people seek for when booking an airbnb: what variables have the largest impact on a listing ?
* How are successful listings distributed in the towns of interest ?
* What can be learned from the "big players" and "international players" of the Airbnb platform?

# Datasets
The Airbnb datasets we will use are accessible and downloadable [here](http://insideairbnb.com/index.html). Inside Airbnb is an independent non-commercial website that groups Airbnb insides and allows to explore via some virtual tools how Airbnb is used in 101 cities around the world. 
For each city the main collected informations are:
* detailed listing data
(host information, type of lodgement, type of beds, provided services, price and fees, ...)
* detailed calendar data
* detailed calendar data
* detailed review data
* summary information and metrics for listings, whis is good for visualisations
* summary review data and listing ID
* neighbourhood list for geo filter, sourced from city or open source GIS files
* GeoJSON file of neighbourhoods of the city

In total a list of 106 features is available, without accounting for the geographic informations. 

All the informations about how the data have been collected and pre-processed is available in the source-website. The datasets are highly-comprehensive, very complete and usable.

To provide more insight on the studied town, the previous data set will be completed by public transportation datasets. The ones used in this project are accessible and downloadable [here](https://www.citylines.co/data). This open-source platform is constantly updated and all the data cover several years up to the present days. 
For each city, the main collected informations are:
* transportation systems
* transportation lines
* transportation sections
* transportation section lines
* transportation stations
* transportation modes

All the data are downloaded as .csv files and handled using pandas.

# Code organisation:
The project will be organised by class. Some open sources libraries are used to build very complete methods performing all the steps required in a full pipeline machine learning or visualisation analysis for example. For example, to easily manipulate folium maps and to provide some non-available functionalities, a class is created allowing to easily add different layes, different representation types, to bind colormaps to each layer with the possibility to control their display through buttons, ... An interesting method of the class will also provide the possibility to build geographical time-lapsed animations of data.

To visualise results, several open-source tools are used. Among them we can cite folium, word clouds representations, sankey diagrams, heatmaps, ... 

# A list of internal milestones up until project milestone 2
* A preliminary phase: getting the data. This is done with web scraping scripts. This allows to refresh the entire data-set with the last available version and to easily introduce a nomenclature system for the saved files.

* Data exploration. Small subsets analyses, and establishment of a pipeline allowing to deal with the missing data.

* Defining succes: To define success, the most relevant customer- and host-associated features are taken into considerations. Implementation of some machine learning algorithms (Random Forest) to determine which features are the most useful for answering the problem. We define the success as the combination of a good overall score, a positive review and a frequently rented listing; which will be the predictor features.

* Random Forest modelling: for classification of features importance and for the study the features contribution on the overall prediction. Model parameters tuning, data normalisation or standardisation and other common techniques are considered and tested to come up with the best model.

* Formatting of the text features (the reviews) to sentiment metrics based on words rating. For each reviews, a value is given based on its negativity, positivy and neutrality; the sum of which equals one.

* Supplement the data with external information about the local environment with the public transportation database. The collected data has to be cleaned and merged in an unique data frame from which some interesting features such as the distance of an Airbnb offer from the nearest station or the number of near stations are extracted.

* Learning from the best hosts by an in-depth analysis of the international presence of some hosts and/or of the multi-owners to answer to the following question: can something be learned on how to define a successful Airbnb from those important actors? And are really those international actors offering best services?

# List of tasks for milestone 3
* Include transportations data-sets in out analysis

* Improve RF model, for example with the already implemented hyperparameter tuning or to allow multitarget predictions

* Analyse other cities, go on the international level

* An interesting visual representation would be to show some folium maps, exploiting the already coded classes 



# Questions for TAa
1) For the review analysis we performed some text analysis using the Natural Language Toolkit library. From this analysis we get several metrics: the negativity (NEG), the positivity (POS) and the neutrality (NEUTR) [NEG + POS + NEUTR = 1]. Should we use the neutrality metric as well in our analysis of the airbnb success? In fact this parameter should only provide a neutral contribution...
2) One of the next step will be a multitarget prediction. This will allow us to build a model with with several target variables considered at once. Do you have any general suggestion to proceed efficiently?
3) We discovered that if the number of reviews of month is used as target variable in the RF model, the accuracy is very low. Do you think this is still a good metric and that we just have to ameliorate the model?
4) Do you think that our approach to answer to our research question is in general good?