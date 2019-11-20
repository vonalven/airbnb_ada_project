# What characterises Airbnb around the world?

![](./img/airbnb-part-de-marche-paris.jpg)
(Image retrieved from: https://pagtour.info/on-evoque-souvent-airbnb-mais-rarement-les-problemes-des-clients/)

# Abstract

In this project, we would like to determine what makes a successful Airbnb housing. 
To do so, we will be working with the data sets provided by the website http://insideairbnb.com/index.html. On this website are datasets for a hundred cities across the world. These datasets provide information about every airbnb housing in these cities, their host's name, the location (geographic coordinates and the neighbourhood), the rooms type, the price and more. 

A first phase of the research would be to focus on the data from the most visited cities in Europe. These are : 
1.  London    
2.  Paris    
3.  Istanbul    
4.  Roma    
5.  ... 


In this section we would like to assess the popularity of airbnb as well as its impact on the housing. We expect this to be achieved by looking at the localisation and amount of airbnbs homes but also how many days per year are these homes for rent for example. 

A second phase would be to 
1. extend the analyses to:
	* Very touristic cities outside of Europe to see if these results could be generalised to an international level.
	* Large-scale metropoles to see if the city size has an influence on airbnb stays.
2. study in which measure the distance from natural morphological point of interest (mountains, sea, ...) affects the prices.
3. study ownership heterogeneity in cities.

# Research questions
Through these analyses we expect to answer the questions:
* Under which conditions is a city likely to have high concentration of airbnb housings? 
* Which criteria influence the renting rate and the success of an airbnb?
* How are airbnb housings geographically distributed, has airbnb taken over the renting market in some Neighbourhood ? 


# Dataset
The datasets we will use are accessible and downloadable here: http://insideairbnb.com/index.html

Inside Airbnb is an independent non-commercial website that groups Airbnb insides and allows to explore via some virtual tools how Airbnb is used in the world. The datasets for each cities are provided and accessible and have already been used by some data-scientist to create some data-stories. For each city the following data are present:
* detailed listing data
* detailed calendar data
* detailed calendar data
* detailed review data
* summary information and metrics for listings -> good for visualisations
* summary review data and listing ID
* neighbourhood list for geo filter, sourced from city or open source GIS files
* GeoJSON file of neighbourhoods of the city

All the informations about how the data have been collected and pre-processed is available in the source-website. The datasets are highly-comprehensive, very complete and usable. They have in fact been organised in a way easily usable by the virtual visualisation tools available on the source website. We therefore expect a relatively easy cleaning procedure.
The datasets we will use are in .csv format and well formatted. They are therefore easy to upload and handle. In a first phase a file-organisation will be performed in order to group the needed informations under one unique DataFrame for easy access. An exploratory analysis followed by cleaning will follow.

# A list of internal milestones up until project milestone 2
1. Data loading, merging. Setting up the right environment needed for the subsequent analyses.
2. Data exploration and cleaning.
3. Advance with the proposed analyses with the goal to generate useful informations that will allow to answer the listed research questions. The goal for milestone 2 is to generate almost every figure so that for milestone 3 we can concentrate about the creation of a data story and the presentation of our results.

# Questions for TAa
...