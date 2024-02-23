import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

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

def scoreViaVectorMethod(df,vectorObject,queries,maxVisibleCandidates):
    job_titles = df[['job_title']].values.ravel()
    vectorList = np.array(list(job_titles) + list(queries))
    allVectors = vectorObject.fit_transform(vectorList)
    candidateVectors = vectorObject.transform(job_titles)
    for query in queries:
        print(query)
        candidates = df
        queryVector = vectorObject.transform(np.array([query]))
        candidates['fit'] = cosine_similarity(candidateVectors,queryVector)
        sortAndDisplay(candidates,maxVisibleCandidates)

def sortAndDisplay(df,maxVisibleCandidates):
    df = df.sort_values(by=['fit','connection'],ascending=False)
    df = df.reset_index(drop=True)
    for i in range(maxVisibleCandidates):
        print(f"{i+1}: {df.loc[i,'job_title']} {df.loc[i,'fit']}")
    print('---')

def sortAndReturnSortedDF(df):
    df = df.sort_values(by=['fit','connection'],ascending=False)
    df = df.reset_index(drop=True)
    return df

def embedding_for_vocab(filepath, word_index,
                        embedding_dim):
    vocab_size = len(word_index) + 1
     
    # Adding again 1 because of reserved 0 index
    embedding_matrix_vocab = np.zeros((vocab_size,
                                       embedding_dim))
 
    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix_vocab[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
 
    return embedding_matrix_vocab

def getGloveVector(embedding_matrix_vocab,word,tokenizer):
    return embedding_matrix_vocab[tokenizer.word_index[word]]

def retrieveAverageGloveVectorFromTitle(title,embedding_matrix_vocab,tokenizer):
    recognizedWords = [word.lower() for word in title.split() if (word.lower() in tokenizer.word_index)]
    return np.mean(np.array([getGloveVector(embedding_matrix_vocab,word,tokenizer) for word in recognizedWords]),axis=0)

def retrieveGloveSimilarityScore(title,query,embedding_matrix_vocab,tokenizer):
    titleVector = retrieveAverageGloveVectorFromTitle(title,embedding_matrix_vocab,tokenizer)
    queryVector = retrieveAverageGloveVectorFromTitle(query,embedding_matrix_vocab,tokenizer)
    return cosine_similarity(queryVector.reshape(1,-1), titleVector.reshape(1,-1))[0,0]

def saveModel(model,modelName):
    pickle.dump(model, open(f"../Models/{modelName}.pkl", 'wb'))
    
