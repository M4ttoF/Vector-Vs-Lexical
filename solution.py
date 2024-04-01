# Matthew Farias

import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from numpy import vectorize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import pytrec_eval
from heapq import nlargest


# Downloading corpus
nltk.download('brown')

# Get 10 most similar words with tf-idf
def tfidf10(term, corpus):
    vectorizer = TfidfVectorizer()
    corpusLower = [x.lower() for x in corpus]
    x = vectorizer.fit_transform(corpusLower) # Fit on vector and Lowercase

    corpusSet = set(corpusLower)
    tfidfList = {}

    for word in term:
        if word not in corpusSet:
            continue
        sim = []
        for word_ in term:
            if word == word_ or word_ not in corpusSet:
                continue
            vectors = vectorizer.transform([word, word_])
            sim.append((cosine_similarity(vectors[0], vectors[1])[0][0], word_))
        largest = nlargest(10, sim)
        tfidfList[word] = {x[1]: x[0] for x in largest}

    return tfidfList

# Get 10 most similar with
def w2v(vocab, corpus, vecSize, window):
    
    tok_corpus = [word_tokenize(s.lower()) for s in corpus]  # Tokenize and lowercase
    wv_model = Word2Vec(sentences=tok_corpus, vector_size=vecSize, window=window, min_count=1, epochs=2000, workers=10)

    w2vList = {}

    for word in vocab:
        if word in wv_model.wv:
            sim = []
            for word_ in vocab:
                if word == word_:
                    continue
                sim.append((cosine_similarity([wv_model.wv[word]], [wv_model.wv[word_]])[0][0], word_))
            largest = nlargest(10, sim)
            w2vList[word] = {x[1]: x[0] for x in largest}

    return w2vList


# Getting top 10 similar from simlex
simDict = {}
with open('SimLex-999.txt', 'r') as f:
    next(f)  # Skip the header line
    for line in f:
        arr = line.strip().split('\t')
        word1, word2, similarity = arr[0],arr[1],arr[3]
        if word1 not in simDict:
            simDict[word1] = {}
        if word2 not in simDict:
            simDict[word2] = {}
        simDict[word1][word2] = float(similarity)
        simDict[word2][word1] = float(similarity)

simlex10={}
for word in simDict:
    simlex10[word]= nlargest(10, simDict[word].items(), key=lambda x: x[1])



from statistics import mean

def stats(query, relevance_results):
    mapScores = [result["map"] for result in relevance_results.values()]
    ndcgScores = [result["ndcg"] for result in relevance_results.values()]
    
    mapAvg = mean(mapScores)
    ndgcAvg = mean(ndcgScores)
    
    return mapAvg, ndgcAvg





for corp in ["adventure", "romance"]:
   print("tfidf", corp)
   print(stats(simlex10, tfidf10(simlex10, brown.words(categories=[corp]))))
   for vSize in (10,50,100,300):
       for wSize in (1,2,5,10):
        print("word2vec", corp, "vector size: ", vSize, ", window size: ", wSize)
        print(stats(simlex10, w2v(simlex10, brown.words(categories=[corp]), vSize, wSize)))


