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

def saveDB(obj, name):
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)
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
initDB2(infoUrl2018icmleca, allPapers, "icml2018eca")
initDB2(infoUrl2018icmlnampi, allPapers, "icml2018nampi")
initDB2(infoUrl2018icmlrml, allPapers, "icml2018rml")
initDB2(infoUrl2018nips1, allPapers, "nips2018cdnnria")
initDB2(infoUrl2018nips2, allPapers, "nips2018mloss")
htmlparse.initDB_nips(infoUrl2017nips, allPapers, "nips2017") #slow
htmlparse.initDB_nips(infoUrl2016nips, allPapers, "nips2016")
htmlparse.initDB_icml(infoUrl2018icmlconf, allPapers, "icml2018")
#Print aggregate dataset statistics
print("All Total", len(allPapers.items()))
print("All Non-rejects", len([a for a in allPapers.items() if a[1][1] != "Reject"]))
print("All Accepts", len([a for a in allPapers.items() if "Accept" in a[1][1]]))


saveDB(allPapers, 'paperdata')

