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

#Enable logging for GenSim
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#For use when two URLs needed (one for paper info, one for accept/reject)
def initDB1(infoUrl, commUrl, fullDB, venue):
    infoResp = requests.get(infoUrl)
    commResp = requests.get(commUrl)
    infoDB = json.loads(infoResp.text)
    commDB = json.loads(commResp.text)
    for entry in infoDB["notes"]:
        fullDB[entry["forum"]] = [entry["content"], "", venue]
    for entry in commDB["notes"]:
        if entry["forum"] in fullDB:
            fullDB[entry["forum"]][1] = entry["content"]["decision"]
            #print(fullDB[entry['forum']][0]['title'])
            fullDB[fullDB[entry['forum']][0]['title']] = fullDB[entry['forum']]
            del fullDB[entry['forum']]

#For use when only paper info URL given (all papers accepted)
def initDB2(infoUrl, fullDB, venue):
    infoResp = requests.get(infoUrl)
    infoDB = json.loads(infoResp.text)
    for entry in infoDB["notes"]:
        fullDB[entry["content"]['title']] = [entry["content"], "Accept", venue]

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
        print("power iteration #%d" % i)
        prev_scores = scores
        scores = (alpha * (scores * X + np.dot(dangle, prev_scores))
                  + (1 - alpha) * prev_scores.sum() / n)
        # check convergence: normalized l_inf norm
        scores_max = np.abs(scores).max()
        if scores_max == 0.0:
            scores_max = 1.0
        err = np.abs(scores - prev_scores).max() / scores_max
        print("error: %0.6f" % err)
        if err < n * tol:
            return scores

    return scores
        
# For ICLR 2017 Conference
# Cross check with stats: https://iclr.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2017:ranzato_introduction_iclr2017.pdf
infoUrl2017Conf = "https://openreview.net/notes?invitation=ICLR.cc%2F2017%2Fconference%2F-%2Fsubmission"
commUrl2017Conf = "https://openreview.net/notes?invitation=ICLR.cc%2F2017%2Fconference%2F-%2Fpaper.*%2Facceptance"

#For ICLR 2017 Workshop
infoUrl2017WS = "https://openreview.net/notes?invitation=ICLR.cc%2F2017%2Fworkshop%2F-%2Fsubmission"
commUrl2017WS = "https://openreview.net/notes?invitation=ICLR.cc%2F2017%2Fworkshop%2F-%2Fpaper.*%2Facceptance"

# For ICLR 2018 Conference
infoUrl2018Conf = "https://openreview.net/notes?invitation=ICLR.cc%2F2018%2FConference%2F-%2FBlind_Submission&details=replyCount&offset=0&limit=1000"
commUrl2018Conf = "https://openreview.net/notes?invitation=ICLR.cc%2F2018%2FConference%2F-%2FAcceptance_Decision&noDetails=true&details=replyCount%2Ctags&offset=0&limit=1000"

# For ICLR 2018 Workshop
infoUrl2018WS = "https://openreview.net/notes?invitation=ICLR.cc%2F2018%2FWorkshop%2F-%2FSubmission&details=replyCount%2Ctags&offset=0&limit=1000"
commUrl2018WS = "https://openreview.net/notes?invitation=ICLR.cc%2F2018%2FWorkshop%2F-%2FAcceptance_Decision&noDetails=true&details=replyCount%2Ctags&offset=0&limit=1000"

#For NIPS 2018 CDNNRIA
infoUrl2018nips1 = "https://openreview.net/notes?invitation=NIPS.cc%2F2018%2FWorkshop%2FCDNNRIA%2F-%2FBlind_Submission&details=replyCount%2Ctags&offset=0&limit=100"
commUrl2018nips1 = "https://openreview.net/notes?invitation=NIPS.cc%2F2018%2FWorkshop%2FCDNNRIA%2F-%2FPaper.*%2FDecision&details=replyCount%2Ctags&offset=0&limit=100"

#For NIPS 2018 MLOSS
infoUrl2018nips2 = "https://openreview.net/notes?invitation=NIPS.cc%2F2018%2FWorkshop%2FMLOSS%2F-%2FSubmission&details=replyCount%2Ctags&offset=0&limit=50"
commUrl2018nips2 = ""

#For NIPS 2018 IRASL //unfinished reviews

#For NIPS 2018 MLITS //unfinished reviews

#For NIPS 2018 Spatiotemporal //unfinished reviews

#For NIPS 2017 Conference
infoUrl2017nipsconf = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016"

#For NIPS 2017
infoUrl2017nips = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-30-2017"

#For NIPS 2016
infoUrl2016nips = "https://papers.nips.cc/book/advances-in-neural-information-processing-systems-29-2016"

#For ICML 2018 ECA
infoUrl2018icmleca = "https://openreview.net/notes?invitation=ICML.cc%2F2018%2FECA%2F-%2FSubmission&details=replyCount%2Ctags&offset=0&limit=50"

#For ICML 2018 NAMPI
infoUrl2018icmlnampi = "https://openreview.net/notes?invitation=ICML.cc%2F2018%2FWorkshop%2FNAMPI%2F-%2FBlind_Submission&details=replyCount%2Ctags&offset=0&limit=50"

#For ICML 2018 RML
infoUrl2018icmlrml = "https://openreview.net/notes?invitation=ICML.cc%2F2018%2FRML%2F-%2FSubmission&details=replyCount%2Ctags&offset=0&limit=50"

#For ICML 2018 Conference
infoUrl2018icmlconf = "http://proceedings.mlr.press/v80/"
#Initialize DB with all relevant data
allPapers = {}
initDB1(infoUrl2017WS, commUrl2017WS, allPapers, "iclr2017ws")
initDB1(infoUrl2017Conf, commUrl2017Conf, allPapers, "iclr2017conf")
initDB1(infoUrl2018WS, commUrl2018WS, allPapers, "iclr2018ws")
initDB1(infoUrl2018Conf, commUrl2018Conf, allPapers, "iclr2018conf")
#initDB2(infoUrl2018icmleca, allPapers, "icml2018eca")
#initDB2(infoUrl2018icmlnampi, allPapers, "icml2018nampi")
#initDB2(infoUrl2018icmlrml, allPapers, "icml2018rml")
#initDB2(infoUrl2018nips1, allPapers, "nips2018cdnnria")
#initDB2(infoUrl2018nips2, allPapers, "nips2018mloss")
#htmlparse.initDB_nips(infoUrl2017nips, allPapers, "nips2017") #slow
#htmlparse.initDB_nips(infoUrl2016nips, allPapers, "nips2016")
#htmlparse.initDB_icml(infoUrl2018icmlconf, allPapers, "icml2018")
#Print aggregate dataset statistics
print("All Total", len(allPapers.items()))
print("All Non-rejects", len([a for a in allPapers.items() if a[1][1] != "Reject"]))
print("All Accepts", len([a for a in allPapers.items() if "Accept" in a[1][1]]))

#Extract relevant data from dataset, preserving order
titles = [paper[0]['title'] for paper in allPapers.values() if 'Accept' in paper[1]]
abstracts = [paper[0]['abstract'] for paper in allPapers.values() if 'Accept' in paper[1]]
pdfLinks = [paper[0]['pdf'] for paper in allPapers.values() if 'Accept' in paper[1]]
venues = [paper[2] for paper in allPapers.values() if 'Accept' in paper[1]]
documents = [paper[0]['title']+'. '+ ' '.join(paper[0]['keywords']) + '. '+paper[0]['abstract']
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
#vec_lsi = corpus_lsi[0]
#sims = index[vec_lsi]
#sims = sorted(enumerate(sims), key=lambda item: -item[1])
#print(list(enumerate(sims))[0:9])
#print(titles[0])
#for i in range(5):
#    print(titles[sims[i][0]])

#For now, just print out first 5 papers in DB and top 3 recommended papers
for i in range(6):
    print("Title:", titles[i])
    print("Abstract:", abstracts[i],)
    print("PDF:", pdfLinks[i])
    print("Venue:", venues[i], '\n')
    vec_lsi = corpus_lsi[i]
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for j in range(1, 3):
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




