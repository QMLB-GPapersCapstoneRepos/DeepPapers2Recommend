import os
from flask import Flask
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
import sys
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

#Method to load raw data
def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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

    @app.route('/')
    def defaultpage():
        return redirect(url_for('search'))

    @app.route('/search')
    def search():
        if request.method == 'GET':
            query = request.args.get('q', '')
            searchlimit = request.args.get('limit', '')
            app.logger.info(query)
            keywords = set([w.lower() for w in query.split()])

            #Count number of keyword matches in each title/abstract/author combination
            eSum = [sum([1 for w in e if w in keywords]) for e in texts]

            #Count number of keywords which are found; prioritize this metric
            kSum = []
            for e in texts:
                kFound = set()
                for w in e:
                    if w in keywords:
                        kFound.add(w)
                kSum.append(len(kFound))
            
            #Generate list of indices
            indices = [i for i in range(len(texts))]

            #Zip the four lists together to sort by keyword matches (with most matches at top of list)
            res = list(zip(kSum, eSum, texts, indices))
            res = sorted(res, reverse=True)
            searchres = []
                
            for i in range(len(res)):
                if res[i][0] == 0 or i >= len(res):
                    break
                data = {}
                data['title'] = titles[res[i][3]]
                data['authors'] = ', '.join(authors[res[i][3]])
                data['abstract'] = abstracts[res[i][3]]
                data['pdf'] = pdfLinks[res[i][3]]
                searchres.append(data)

            #Add limit with default value of 10
            toshow = min(10, len(searchres))
            if searchlimit != '':
                toshow = min(int(searchlimit), len(searchres))
                
        return render_template('search.html', searchres=searchres, indexref=res, query=query, toshow=toshow)

    @app.route('/recommend')
    def recommend():
        if request.method == 'GET':
            pindex = request.args.get('pindex', '')
            reclimit = request.args.get('limit', '')
            vec_lsi = corpus_lsi[int(pindex)]
            sims = index[vec_lsi]
            sims = sorted(enumerate(sims), key=lambda item: -item[1])
            recLimit = 10
            recs = []
            for j in range(0, len(sims)):
                #Handle openreview pdf links, which do not include domain!
                domain = ''
                pdfLink = pdfLinks[sims[j][0]]
                if pdfLink[0:4] == '/pdf':
                    domain = 'http://www.openreview.net'

                data = {}
                data['score'] = sims[j][1]
                data['title'] = titles[sims[j][0]]
                data['authors'] = ', '.join(authors[sims[j][0]])
                data['abstract'] = abstracts[sims[j][0]]
                data['pdf'] = domain+pdfLinks[sims[j][0]]
                data['venue'] = venues[sims[j][0]]
                
                recs.append(data)

            #Add limit with default value of 10
            #Subtract 1 to not count the same paper
            toshow = min(10, len(recs)-1)
            if reclimit != '':
                toshow = min(int(reclimit), len(recs)-1)
        return render_template('recommend.html', recs=recs, toshow=toshow)
    
    return app

application = create_app()

if __name__ == "__main__":
    application.run()
