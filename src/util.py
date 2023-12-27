import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity

def reScoreSpacySimilarityBasedOnStar(starIndex,df,query):
    starTitle = df.loc[starIndex,'job_title']
    df['fit'] = df['job_title'].apply(lambda x: scoreSpacySimilary(f"{starTitle} {query}",x))
    return df.sort_values(by=['fit','connection'],ascending=False)

def scoreSpacySimilary(target,candidate):
    nlp = spacy.load("en_core_web_lg")
    return nlp(target).similarity(nlp(candidate))

def scoreBagOfWordsSimilarity(bagOfWordsVectorizer,query,x):
    queryVector = bagOfWordsVectorizer.transform([query])
    candidateVector = bagOfWordsVectorizer.transform([x])
    return cosine_similarity(queryVector, candidateVector)