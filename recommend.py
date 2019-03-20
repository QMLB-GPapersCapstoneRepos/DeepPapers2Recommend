import json
import requests
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
import htmlparse
import logging
from scipy import sparse
import numpy as np
from sklearn.decomposition import randomized_svd
from sklearn.externals.six.moves.urllib.request import urlopen
from sklearn.externals.six import iteritems
import pickle

#Enable logging for GenSim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def centrality_scores(X, alpha=0.85, max_iter=100, tol=1e-10):
    """Power iteration computation of the principal eigenvector

    This method is also known as Google PageRank and the implementation
    is based on the one from the NetworkX project (BSD licensed too)
    with copyrights by:

      Aric Hagberg <hagberg@lanl.gov>
      Dan Schult <dschult@colgate.edu>
      Pieter Swart <swart@lanl.gov>
    """
    n = X.shape[0]
    X = X.copy()
    incoming_counts = np.asarray(X.sum(axis=1)).ravel()

    print("Normalizing the graph")
    for i in incoming_counts.nonzero()[0]:
        X.data[X.indptr[i]:X.indptr[i + 1]] *= 1.0 / incoming_counts[i]
    dangle = np.asarray(np.where(np.isclose(X.sum(axis=1), 0),
                                 1.0 / n, 0)).ravel()

    scores = np.full(n, 1. / n, dtype=np.float32)  # initial guess
    for i in range(max_iter):
        if i%10 == 0: print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        if i%10 == 0: print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores

#Method to load raw data
def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)        

#Initialize DB with all relevant data
allPapers = load_obj('paperdata')

#Print aggregate dataset statistics
print("All Total", len(allPapers.items()))
print("All Non-rejects", len([a for a in allPapers.items() if a[1][1] != "Reject"]))
print("All Accepts", len([a for a in allPapers.items() if "Accept" in a[1][1]]))

#Extract relevant data from dataset, preserving order
titles = [paper[0]['title'] for paper in allPapers.values() if 'Accept' in paper[1]]
abstracts = [paper[0]['abstract'] for paper in allPapers.values() if 'Accept' in paper[1]]
pdfLinks = [paper[0]['pdf'] for paper in allPapers.values() if 'Accept' in paper[1]]
authors = [paper[0]['authors'] for paper in allPapers.values() if 'Accept' in paper[1]]
venues = [paper[2] for paper in allPapers.values() if 'Accept' in paper[1]]
documents = [paper[0]['title']+'. '+ ' '.join(paper[0]['keywords']) + '. '+paper[0]['abstract'] + '. '+' '.join(paper[0]['authors'])
             for paper in allPapers.values() if 'Accept' in paper[1]]

#Exclude some common words
stoplist = set('for a of the and to in this we by are our it'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

#Count frequency of each word
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

#Remove tokens which only occur once
#texts = [[token for token in text if frequency[token] > 1]
#         for text in texts]
dictionary = corpora.Dictionary(texts)
#print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=400)
corpus_lsi = lsi[corpus_tfidf]

#Load preprocessed corpus
#corpus_lsi = corpora.MmCorpus('lsicorpus.mm')

#Find similarity matrix
#index = similarities.MatrixSimilarity.load('paperSimMat.pkl')
index = similarities.MatrixSimilarity(corpus_lsi)

flag1 = True
while flag1:
    #Ask user for search query
    query = input("Enter query (q! to quit): ")
    if query == "q!":
        flag1 = False
        break
    #Extract keywords from query
    keywords = set([w.lower() for w in query.split()])

    #Count number of keyword matches in each title/abstract/author combination
    eSum = [sum([1 for w in e if w in keywords]) for e in texts]

    #Generate list of indices
    indices = [i for i in range(len(texts))]

    #Zip the three lists together to sort by keyword matches (with most matches at top of list)
    res = list(zip(eSum, texts, indices))
    res = sorted(res, reverse=True)

    printLimit = int(input("Maximum print number? (enter number) "))
    currIndex = 0
    while currIndex < printLimit and res[currIndex][0] > 0:
        #currIndex += 1
        paperNum = res[currIndex][2]
        #print(paperNum)
        currIndex += 1
        print('Result '+str(currIndex)+': ')
        print('Title:', titles[paperNum])
        print('Authors: ', authors[paperNum])
   # print(res[currIndex][1])
    flag2 = True
    while flag2:
        recIndex = int(input("Which paper? (enter number, <= 0 to quit) "))
        recIndex -= 1
        if recIndex < 0:
            flag2 = False
            break
        #print(recIndex)
        recIndex = res[recIndex][2]
        #print(recIndex)
        #add error catching here

        vec_lsi = corpus_lsi[recIndex]
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        recLimit = int(input("Maximum rec print number? (enter number) "))
        for j in range(0, recLimit+1):
            #Handle openreview pdf links, which do not include domain!
            domain = ''
            pdfLink = pdfLinks[sims[j][0]]
            if pdfLink[0:4] == '/pdf':
                domain = 'www.openreview.net'
            #Excerpt abstract to increase readability
            excerptTail = ''
            excerptLen = min(len(abstracts[sims[j][0]]), 300)
            if excerptLen == 300:
                excerptTail = '...'

            print("Rank", j, titles[sims[j][0]])
            print("Similarity Score:", sims[j][1])
            print("Authors:", authors[sims[j][0]])
            print("Abstract:", abstracts[sims[j][0]][0:excerptLen]+excerptTail)
            print("PDF:", domain+pdfLinks[sims[j][0]])
            print("Venue:", venues[sims[j][0]], '\n')
'''
#For now, just print out first 5 papers in DB and top 3 recommended papers
for i in range(6):
    print("Title:", titles[i])
    print("Abstract:", abstracts[i],)
    print("PDF:", pdfLinks[i])
    print("Venue:", venues[i], '\n')
    vec_lsi = corpus_lsi[i]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for j in range(0, 2):
        print("Similarity Score:", sims[j][1])
        print("Rank", j, titles[sims[j][0]])
        print("Abstract:", abstracts[sims[j][0]])
        print("PDF:", "www.openreview.net"+pdfLinks[sims[j][0]])
        print("Venue:", venues[sims[j][0]], '\n')
    print('---')

threshold = 0.25
#vec_lsi2 = corpus_lsi[0]
#print(vec_lsi2)
#print(len(corpus_lsi))
adj_matrix = sparse.lil_matrix((len(corpus_lsi), len(corpus_lsi)), dtype=np.float32)
for i in range(len(corpus_lsi)):
    vec_lsi = corpus_lsi[i]
    sim_scores = index[vec_lsi]
    for j in range(len(sim_scores)):
        if sim_scores[j] > threshold:
            adj_matrix[i,j] = 1.0#sim_scores[j]
        else:
            adj_matrix[i,j] = 0.0
adj_matrix = adj_matrix.tocsr()
U, s, V = randomized_svd(adj_matrix, 5, n_iter=3)
pprint([titles[i] for i in np.abs(U.T[0]).argsort()[-10:]])
pprint([titles[i] for i in np.abs(V[0]).argsort()[-10:]])
print('-----')
scores = centrality_scores(adj_matrix, max_iter=100, tol=1e-10)
pprint([titles[i] for i in np.abs(scores).argsort()[-10:]])
'''



