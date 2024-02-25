import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import util
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def reScoreBasedOnStar(df,starIndex,query,model):
    job_titles = df[['job_title']].values.ravel()
    model = pickle.load(open("../models/SBERT.pkl", 'rb'))
    job_title_embeddings = model.encode(job_titles)
    
    starTitle = df.loc[starIndex,'job_title']
    oldStarScore = df.loc[starIndex,'fit']

    query_embedding = model.encode(query)
    queryArray = query_embedding.reshape(1,-1)

    star_embedding = model.encode(starTitle)
    starArray = star_embedding.reshape(1,-1)

    queryScore = cosine_similarity(queryArray, job_title_embeddings).reshape(-1,1)
    starScore = cosine_similarity(starArray, job_title_embeddings).reshape(-1,1)
    similarities = ((oldStarScore)*queryScore) + ((1-oldStarScore)*starScore)
    df['fit'] = np.array(similarities)
    return df.sort_values(by=['fit','connection'],ascending=False)

def main():
    candidates = pd.read_csv('../data/raw/potential-talents-AspiringHumanResources-SeekingHumanResources.csv')
    candidates.loc[candidates['connection'] == '500+ ','connection'] = '500'
    candidates['connection'] = candidates['connection'].astype(int)
    #candidates = candidates.drop_duplicates(subset=['job_title','location','connection'])
    job_titles = candidates[['job_title']].values.ravel()
    model = pickle.load(open("../models/SBERT.pkl", 'rb'))
    job_title_embeddings = model.encode(job_titles)

    query = input("input query: ")
    resultFileName = query
    query_embedding = model.encode(query)
    queryArray = query_embedding.reshape(1,-1)
    
    similarities = cosine_similarity(queryArray, job_title_embeddings)
    candidates['fit'] = np.array(similarities).reshape(-1,1)
    candidates = candidates.sort_values(by=['fit','connection'],ascending=False)
    candidates = candidates.reset_index()
    maxVisibleCandidates = int(input("View how many candidates? "))
    for i in range(maxVisibleCandidates):
        print(f"{i}: {candidates.loc[i,'job_title']} {candidates.loc[i,'location']} {candidates.loc[i,'connection']} {candidates.loc[i,'fit']}")

    starDecision = ''
    while (starDecision != 'y') & (starDecision != 'n'):
        starDecision = input('Do you wish to star a candidate? y or n: ')
    if starDecision == 'y':
        starIndex = -1
        while (starIndex < 0) | (starIndex >= maxVisibleCandidates): 
            starIndex = int(input('Which candidate index do you want to star? '))
        resultFileName = (query + " " + candidates.loc[starIndex,'job_title']).replace(" ","_")
        candidates = reScoreBasedOnStar(candidates,starIndex,query,model)
    candidates = candidates.reset_index(drop=True).drop('index',axis=1)
    candidates.to_csv(f'../data/results/{resultFileName}_results.csv')
        

if __name__=='__main__':
    main()