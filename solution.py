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
nltk.download('punkt')
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

# Get 10 most similar with Word2Vec
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

# Getting top 10 similar from SimLex-999
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

# Function to compute average MAP and nDCG
def stats(query_relevance, retrieval_results):
    evaluator = pytrec_eval.RelevanceEvaluator(query_relevance, {'map', 'ndcg'})
    evaluation_results = evaluator.evaluate(retrieval_results)
    map_values = [query_result['map'] for query_result in evaluation_results.values()]
    ndcg_values = [query_result['ndcg'] for query_result in evaluation_results.values()]
    avg_map = sum(map_values) / len(map_values)
    avg_ndcg = sum(ndcg_values) / len(ndcg_values)
    return avg_map, avg_ndcg

for corp in ["adventure", "romance"]:
    print("tfidf", corp)
    simlex_relevance = {term: {word: 1 for word in words} for term, words in simDict.items()}
    retrieval_results = tfidf10(simlex_relevance, brown.words(categories=[corp]))
    print(stats(simlex_relevance, retrieval_results))
    for vSize in (10,50,100,300):
        for wSize in (1,2,5,10):
            print("word2vec", corp, "vector size: ", vSize, ", window size: ", wSize)
            retrieval_results = w2v(simlex_relevance, brown.words(categories=[corp]), vSize, wSize)
            print(stats(simlex_relevance, retrieval_results))
