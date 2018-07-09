from gensim import models
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
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
#from sklearn.manifold import tsne
import tsne
from tsne import bh_sne


File = open("units") #open file
units = File.readlines() #read all lines
File.close()

units = [x.strip() for x in units]

def intersect(a, b):
    return not set(a).isdisjoint(b)

def document_vector(doc):
    vec = np.zeros(11)
    sentences = nltk.sent_tokenize(doc)
    verb = 1
    lemmatizer = WordNetLemmatizer()
    all_tokens = []
    all_entities = []
    all_agents = []
    all_tenses = []
    for sentence in sentences:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
	    all_tokens.append(word)
	    if ('VB' in pos):
		if (pos == 'VBD' or pos == 'VBN'):
		    all_tenses.append('past')
		else:
		    all_tenses.append('present')
		lem = lemmatizer.lemmatize(word, pos='v')
		if (lem != 'be'):
		    verb = 0
		if (lem == 'want' or lem == 'require' or lem == 'need'):
		    vec[1] = 1
	    if (pos == 'NNS'):
		lem = lemmatizer.lemmatize(word, pos='n')
		#print lem
		if (wordnet.synsets(lem)):
		    all_entities.append(wordnet.synsets(lem)[0])
	    if (pos == 'NNP' or pos == 'NNPS'):
	        all_agents.append(word)
	    if (word in ['each', 'equally', 'per', 'every']):
		vec[6] = 1
	    if (word == 'left'):
		vec[7] = 1
	    if (word in ['altogether', 'together']):
		vec[8] = 1
	    if (word in ['less','more']):
		vec[9] = 1
	    if (word == 'cost'):
		vec[10] = 1
    if ('as many as' in sentence):
	vec[6] = 1    
    if ('in all' in sentence):
	vec[8] = 1
    if ('er than' in sentence):
	vec[9] = 1

    for entity in all_entities:
	hyperentity = set([i for i in entity.closure(lambda s:s.hypernyms())])
	if (intersect(hyperentity, all_entities)):
	    vec[1] = 1
	    continue
    for token in all_tokens:
	antonyms = []
	for syn in wordnet.synsets(token):
	    for l in syn.lemmas():
		if (l.antonyms()):
		    for ant_l in l.antonyms():
			#print ant_l.name()
		    	antonyms.append(ant_l.name())
        #print antonyms
        if (intersect(antonyms, all_tokens)):
	    vec[1] = 1
            continue
    if (intersect(units, all_tokens)):
	vec[2] = 1
    if (len(all_agents) > 1):
	vec[3] = 1
    if (len(all_entities) > 1):
	vec[4] = 1
    all_tenses = set(all_tenses)
    #print(all_tenses)
    if (len(all_tenses) > 1):
	vec[5] = 1
    vec[0] = verb
    return vec

print document_vector("There are 5 apples on the table. 20 fruits are on the table. One apple weighs 2 gram. Mary ate an apple.")

def preprocess(text):
    text = text.lower()
    return text

File = open("q2.txt") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]



f1 = open("q2-schema")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))
    #print i.rstrip() + '|' + `l[counter]`;
#print(intoutput)


y = []
for i in range(0, len(data)):
    y.append(document_vector(preprocess(data[i])))

print(len(y))

'''
def train_classifier(X,y):
    n_estimators = [200,400]
    min_samples_split = [2]
    min_samples_leaf = [1]
    bootstrap = [True]

    parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
                  'min_samples_split': min_samples_split}

    clf = GridSearchCV(RFC(verbose=1,n_jobs=4), cv=4, param_grid=parameters)
    clf.fit(X, y)
    return clf

X_train, X_test, y_train, y_test = cross_validation.train_test_split(y, intoutput, test_size=0.2, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))

avg = 0.0

#print(len(y[0]))
def purity_score(clusters, classes):
    """
    Calculate the purity score for the given cluster assignments and ground truth classes
    
    :param clusters: the cluster assignments array
    :type clusters: numpy.array
    
    :param classes: the ground truth classes
    :type classes: numpy.array
    
    :returns: the purity score
    :rtype: float
    """
    
    A = np.c_[(clusters,classes)]

    n_accurate = 0.

    for j in np.unique(A[:,0]):
        z = A[A[:,0] == j, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])

    return n_accurate / A.shape[0]

for j in range(0, 100):
    arrY = np.asarray(y)
    kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=100)  
    X = kmeans_model.fit(arrY)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(arrY)
    comp = [x - 1 for x in intoutput]
    
    #computing purity
    #3 clusters in l and 3 classes in output 
    #Step 1 - identifying N
    N = len(intoutput)
    #Step 2 - aggregating each cluster
    matrix = [[0 for x in range(3)] for m in range(3)] 
    matrix = numpy.zeros_like(matrix)
    for i in range(0, len(labels)):
    #print(`labels[i]` + "|" + `intoutput[i]`)
        matrix[labels[i]][intoutput[i]-1] += 1
    num = 0
    for i in range(0, 3):
        num += max(matrix[i])
    
    purity = purity_score(l, comp)
    avg = avg + purity
avg = avg/100
print(avg)
'''

x_data = np.asarray(y)
#x_data = x_data.reshape((x_data.shape[0], -1))
print(x_data[0])
X_2d = bh_sne(x_data)
label1 = ["", "#FFFF00", "#008000", "#0000FF"]
color = [label1[i] for i in intoutput]
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()

arrY = np.asarray(y)
kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=100)  
X = kmeans_model.fit(arrY)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(arrY)
label1 = ["#FFFF00", "#008000", "#0000FF"]
color = [label1[i] for i in l]
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()
    
