# What makes an Airbnb successful?

![](./img/airbnb-part-de-marche-paris.jpg)
(Image retrieved from: https://pagtour.info/on-evoque-souvent-airbnb-mais-rarement-les-problemes-des-clients/)

# Abstract

Airbnb is a world-wide known online lodging marketplace which is well established in most of the principal touristic destinations. Since its foundation in 2008, the company experienced a continuous growth with a boom after 2012, ending up to a revenue of 2.6 billions US dollars in 2017, 12736 employees in 2019 and an astonishing 2 millions peoples lodging with Airbnb each night in October of this year. The online platform provides a wide range of services and a variety of lodging experiences. In addition to the classic services, a full set of entertainment propositions ranging from touristic tours, culinary experiences, informations about the local environment and neighbourhoods are proposed. In the latest years Airbnb has often been targeted by some legal process and was put at the center of local scandals related to over-presence causing discriminations in the renting real estates marketplaces. 

In this project, we would like to determine what makes a successful Airbnb housing. The definition of success can be hard to be defined and can have several interpretations. Depending on the specific contest and on the users experiences and expectations, success could be for example the quality of the provided services inside the rented lodge or simply the quality of the experiences outside the residence in the local environment or the proposed price and fees. According to the Cambridge dictionary, success is defined as "something that achieves positive results". For our analysis, we will put ourselves in the shoes of both the customers and the hosts. For hosts one could imagine that this translates in a good evaluation of the proposed services. On the opposite side, customers additionally could demonstrate the their appreciation through a very positive review. 

A lot of informations are constantly collected by various entities from the Airbnb platform through websites scraping. Our analysis is performed with the data sets provided by the website http://insideairbnb.com/index.html. Here, a very exhaustive list of monthly-updated features are available for 101 cities around the world. The data covers a wide temporal range and some informations are detailed through the years. The main features of those collections of data are the host informations (name, nationality, profile pictures, ...), the listings details (type of lodgement, type of beds, provided services, price and fees, ...), the evaluation parameters for each listing, informations about the listings neighbourhoods with geographical data as well and a complete list of customers rentals, reviews and many other informations. In total a list of 106 features is available, without accounting for the geographic informations. 

In a first phase, data exploration is performed. To do this, small subsets are analysed to be computationally efficient and a pipeline allowing to deal with the missing data is established. Some missing information will be deleted and some others inferred, according to the personal thought and experience, some existing literature, the informations provided by the insideairbnb platform about the used criteria and technique to collect  the data and create the features and according to some additional data-driven analyses. 

Once the data set is fully usable, the previously exposed research question is taken into consideration. To define success, the most relevant customer- and host-associated features are taken into considerations. Here some machine learning algorithms are used to determine which features are the most useful for answering our question. We define the success as the combination or the individual value of the scoring parameters of each listing; which will be the predictor features. Those are the overall rating score and the cleanliness, checkin, communication, location and value scores. We judged those features as good candidates for the evaluation of a listing success because those scores directly reflects the overall user experience and also the host effort in providing a good service.

Random Forest modelling is used to classify features importance and to study the influence of features on the overall prediction. This can be useful for example to compare how the weight of different features changes across different neighbourhoods or cities. Model parameters tuning, data normalisation or standardisation and other common techniques are considered and tested to come up with the best problem modelling option.
In this first phase, results could be quite obvious. However, the innovative and exciting results will come from the suggested comparison of the features relevance differences between different data subsets. 

Machine learning or a simpler technique will also be applied to analyse the text-formatted features, such as for example the customers reviews.

In a second phase, it will be interesting to learn from the best hosts. An in-depth analysis of the international presence of some hosts and/or of the multi-owners will bring us to answer to the following question: can something be learned on how to define a successful Airbnb from those important actors? And are really those international actors offering best services? One could in fact expect that owning several listings will provide a greater experience allowing the provision of a better service.

The general idea of this project is to provide an interesting analysis and to expose it with impacting graphical representations. The main goal is to provide a simple and user-friendly overview of the extracted results. To visualise results, several open-source tools are used. Among them we can cite folium, word clouds representations, sankey diagrams, heatmaps, ... 

# Research questions
The summarised research questions are:
* What defines a successful Airbnb?
* How the important parameters affecting success changes when specifically analysing some subsets such as individual neighbourhoods?
* What can be learned from the "big players" and "international players" of the Airbnb platform?

# Dataset
A brief summary of the previous descriptions of the dataset is provided here.
The datasets we will use are accessible and downloadable here: http://insideairbnb.com/index.html. Inside Airbnb is an independent non-commercial website that groups Airbnb insides and allows to explore via some virtual tools how Airbnb is used in the world. The datasets for each cities are provided and accessible and have already been used by some data-scientist to create some data-stories. For each city the main collected informations are:
* detailed listing data
* detailed calendar data
* detailed calendar data
* detailed review data
* summary information and metrics for listings -> good for visualisations
* summary review data and listing ID
* neighbourhood list for geo filter, sourced from city or open source GIS files
* GeoJSON file of neighbourhoods of the city

All the informations about how the data have been collected and pre-processed is available in the source-website. The datasets are highly-comprehensive, very complete and usable. They have in fact been organised in a way easily usable by the virtual visualisation tools available on the source website. A relatively easy cleaning procedure is then implemented. The datasets are .csv files and well formatted. 

# Code organisation:
The project will be organised by class. Some open sources libraries are used to build very complete methods performing all the steps required in a full pipeline machine learning or visualisation analysis for example. For example, to easily manipulate folium maps and to provide some non-available functionalities, a class is created allowing to easily add different layes, different representation types, to bind colormaps to each layer with the possibility to control their display through buttons, ... 

# A list of internal milestones up until project milestone 2
1. Data loading, merging. Setting up the right environment needed for the subsequent analyses.
2. Data exploration and cleaning.
3. Data analysis and models building

# Questions for TAa
...