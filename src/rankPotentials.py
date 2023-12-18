import pandas as pd
import spacy

def reScoreBasedOnStar(starIndex,df,query):
    starTitle = df.loc[starIndex,'job_title']
    df['fit'] = df['job_title'].apply(lambda x: scoreSimilary(f"{starTitle} {query}",x))
    return df.sort_values(by=['fit','connection'],ascending=False)

def scoreSimilary(target,candidate):
    nlp = spacy.load("en_core_web_lg")
    return nlp(target).similarity(nlp(candidate))

def main():
    candidates = pd.read_csv('../data/raw/potential-talents-AspiringHumanResources-SeekingHumanResources.csv')
    candidates.loc[candidates['connection'] == '500+ ','connection'] = '500'
    candidates['connection'] = candidates['connection'].astype(int)
    #candidates = candidates.drop_duplicates(subset=['job_title','location','connection'])

    query = input("input query: ")
    resultFileName = query
    maxVisibleCandidates = int(input("View how many candidates? "))
    candidates['fit'] = candidates['job_title'].apply(lambda x: scoreSimilary(query,x))
    candidates = candidates.sort_values(by=['fit','connection'],ascending=False)
    candidates = candidates.reset_index()
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
        candidates = reScoreBasedOnStar(starIndex,candidates,query)
    candidates = candidates.reset_index(drop=True).drop('index',axis=1)
    candidates.to_csv(f'../data/results/{resultFileName}_results.csv')
        

if __name__=='__main__':
    main()