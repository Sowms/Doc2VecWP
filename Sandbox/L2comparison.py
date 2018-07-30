from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import numpy
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import numpy as np
from pprint import pprint
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))

def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    #doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc


def train(datafile):
    File = open(datafile) #open file
    data = File.readlines() #read all lines
    File.close()
    data = [x.strip() for x in data]
    docLabels = []
    for counter in range(0, len(data)):
        docLabels.append('wp' + `counter`)
    data = nlp_clean(data)
    it = LabeledLineSentence(data, docLabels)
    model = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
    model.build_vocab(it)
    #training of model
    for epoch in range(10):
        print 'iteration'+str(epoch+1)
    model.train(it, total_examples = len(data), epochs = 10)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    #saving the created model
    model.save('doc2vec.model')
    y = []
    for i in range(0,len(data)):
        y.append(model.docvecs[docLabels[i]])
    return y

def calcDistances(vectors):
    distances = []
    for i in xrange (0, len(vectors), 2):
        distances.append(numpy.linalg.norm(vectors[i] - vectors[i+1]))
    return distances

def calcSimDistances(vectors):
    distances = []
    indices = [[1,2],[0,3],[10,13],[11,12],[14,17],[15,16]]
    for i in xrange (0, len(indices)):
        distances.append(numpy.linalg.norm(vectors[indices[i][0]] - vectors[indices[i][1]]))
    return distances

def apply(datafile):
    model = Doc2Vec.load('doc2vec.model')
    File = open(datafile) #open file
    data = File.readlines() #read all lines
    File.close()
    data = [x.strip() for x in data]
    data = nlp_clean(data)
    y = []
    for i in range(0, len(data)):
	y.append(model.infer_vector(data[i]))
    return y
   

plt.figure
pltbarwidth = 0.15
train('illinois-questions')
y1 = calcDistances(apply('sandbox.txt'))
xcoord1 = [(i+1) for i in range(0, len(y1))]
xcoord2 = [(i+1)+pltbarwidth for i in range(0, len(y1))]
xcoord3 = [(i+1)+2*pltbarwidth for i in range(0, len(y1))]
train('correlated-illinois-questions')
y2 = calcDistances(apply('correlated-sandbox'))
#y3 = calcDistances(train('labelled-sandbox'))
conc = []
conc.append(y1)
conc.append(y2)
#conc.append(y3)

objects = ['Pair{0}'.format(i) for i in range(0, len(y1))] 

plt.bar(xcoord1, y1, pltbarwidth, align='center', color = '#6495ed', label='Original')
plt.bar(xcoord2, y2, pltbarwidth, align='center', color = '#ff5733', label='Correlated Annotation')
#plt.bar(xcoord3, y3, pltbarwidth, align='center', color = '#4e972a', label='Complete Annotation')
plt.title('L2 Distances - Dissimilar')
plt.xticks(xcoord1, objects)
plt.legend()
plt.ylabel('Usage')
#n, bins, patches = plt.hist(y1, bins=xcoord)
#plt.ylim(0.035,0.05)

#plt.plot(bins)
'''
plt.scatter(xcoord, y1, color = '#6495ed')
plt.scatter(xcoord, y2, color = '#ff5733')
plt.scatter(xcoord, y3, color = '#4e972a')

'''
plt.show()

plt.figure
pltbarwidth = 0.15
train('illinois-questions')
y1 = calcSimDistances(apply('sandbox.txt'))
xcoord1 = [(i+1) for i in range(0, len(y1))]
xcoord2 = [(i+1)+pltbarwidth for i in range(0, len(y1))]
#xcoord3 = [(i+1)+2*pltbarwidth for i in range(0, len(y1))]
train('correlated-illinois-questions')
y2 = calcSimDistances(apply('correlated-sandbox'))
#y3 = calcDistances(train('labelled-sandbox'))
conc = []
conc.append(y1)
conc.append(y2)
#conc.append(y3)

objects = ['Pair{0}'.format(i) for i in range(0, len(y1))] 

plt.bar(xcoord1, y1, pltbarwidth, align='center', color = '#6495ed', label='Original')
plt.bar(xcoord2, y2, pltbarwidth, align='center', color = '#ff5733', label='Correlated Annotation')
#plt.bar(xcoord3, y3, pltbarwidth, align='center', color = '#4e972a', label='Complete Annotation')
plt.title('L2 Distances - Similar')
plt.xticks(xcoord1, objects)
plt.legend()
plt.ylabel('Usage')
#n, bins, patches = plt.hist(y1, bins=xcoord)
#plt.ylim(0.6,3.1)

#plt.plot(bins)
'''
plt.scatter(xcoord, y1, color = '#6495ed')
plt.scatter(xcoord, y2, color = '#ff5733')
plt.scatter(xcoord, y3, color = '#4e972a')

'''
plt.show()






