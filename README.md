# scraping
Tool for scraping imdb movies data.  
There are five files:  
1. scraping_movie_page : scraping all wanted data from a specific movie page in IMDB with a given url. Inculding scraping rating function to scrap from all movies in already existing data base.
2. scraping_person_page: scraping all wanted data from a specific person page in IMDB with a given url
3. scraping_movies_by_year: getting a list of movie pages urls from a IMDB search and scraping all the movies with scraping_movie_page functions.
4. scraping_all_people: creating a list of all people urls from an existing movie data base and scraping data with those urls with scraping_person_page functions
5. scraping_money_data: scraping movie gross data from "the numbers" website for already existing movies

# DB_API
A module with a lot of functions to create a numeric vectors out of raw data.  
Including functions to combine people data with movie data, converting genres to a binary vector, functions to reduce data samples based on data quality etc.

# Learning
The main module with all the learning experiments using different algorithms and double learning with rating. 
