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
from gensim.test.utils import get_tmpfile
#Enable logging for GenSim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
#print(documents[0])
#print(len(documents))

#Exclude some common words
stoplist = set('for a of the and to in this we by are our it'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]
#print(texts[0])

#Count frequency of each word
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

#Remove tokens which only occur once
texts = [[token for token in text if frequency[token] > 1]
         for text in texts]

dictionary = corpora.Dictionary(texts)
#print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in texts]

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=400)
corpus_lsi = lsi[corpus_tfidf]
#lsi.print_topics(20)
#print(corpus_lsi[0], max(corpus_lsi[0], key=lambda x:x[1]))
#print(corpus_lsi[1], max(corpus_lsi[0], key=lambda x:x[1]))

index = similarities.MatrixSimilarity(corpus_lsi)

#output_fname = get_tmpfile('lsicorpus.mm')
#index.save('paperSimMat.pkl')
corpora.MmCorpus.serialize('lsicorpus.mm', corpus_lsi)
'''
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
        print("what", sims[j][0])
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



