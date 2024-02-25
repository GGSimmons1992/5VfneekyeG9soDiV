# Potential Talents
## Background
This project uses experiments with Bag of Words, TF-IDF, Word2Vec, Glove, FastText, BERT, and SBERT to rank and sort candidates based on how thier job title is similar to the hiring manager's search query. After deciding that SBERT was the best model, we packaged an SBERT model that can be used by util/rankPotentials.py. rankPotentials.py asks the user to enter a query and how many top candidates they want to see on the screen. It will then proceed to ask the user if they want to star a particular visible candidate, and it will re-calculate based off of the starred candidate.

Attributes:

id : unique identifier for candidate (numeric)

job_title : job title for candidate (text)

location : geographical location for candidate (text)

connections: number of connections candidate has, 500+ means over 500 (text)

## data
data is not pushed due to gitignore. Raw data from Apziva is in raw folder. Final refinements will be saved in the results folder. Glove zip folder and txt files are also in this folder.

## models
Holds the finalized SBERT.pkl package

# notebooks
Holds the eda.ipynb notebook which shows the experiments with Bag of Words, TF-IDF, Word2Vec, Glove, FastText, BERT, and SBERT. SBERT is determined as final model.

## src/util
Contatins functions used by both eda.ipynb and rankPotentials.py

## src/rankPotentials
Running rankPotentials.py in src will inquire inputs from the user for a query, how many candidates desired to be seen on the screen, and if they want to star a particular top to refine the search. Final refinements will be saved in the results folder

## Requirements.txt
list of python packages used by this repo.

## LICENSE
This project uses and MIT license