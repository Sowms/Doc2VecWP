from gensim import models
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
from sklearn import metrics
import numpy as np
from pprint import pprint
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from sklearn.neighbors import NearestNeighbors
from ndcg import ndcg_at_k

glove_file = 'glove.6B.100d.txt'
tmp_file = get_tmpfile("test_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = preprocess(doc)
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

def preprocess(text):
    text = text.lower()
    doc = nltk.tokenize.word_tokenize(text)
    #doc = [word for word in doc if word not in stop_words]
    return doc

File = open("questions") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]


y = []
for i in range(0,len(data)):
    y.append(document_vector(model, data[i]))
    
print(len(y))

outfile = open("avgword2vec_model","wb")
pickle.dump(y, outfile)
outfile.close()

File = open("schemas") #open file
schemas = File.readlines() #read all lines
File.close()


def calcAccuracy(neighbours, wpIn):
    k = len(neighbours[0])
    schema = schemas[wpIn]
    match = 0.0
    vec = []
    for i in range(1, k):
	if (schemas[neighbours[0][i]] == schema):
	    match += 1
	    vec.append(5)
        else :
	    vec.append(0)
    return match/(k-1), vec


y_true = []
y_pred = [1] * 9
for j in range(2,11):
    knn = NearestNeighbors(n_neighbors=j)
    knn.fit(y)
    NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=j, p=2, radius=1.0)
    avg = 0.0
    '''
    avg_n = 0.0
    if (j == 11) 
       y_true = vec
       ndcg = metrics.ndcg_score(y_true, y_pred, k=j)
       avg_n = avg_n + ndcg
    '''
    for i in range(0, len(y)):
        neighbours = knn.kneighbors(np.array(y[i]).reshape(1,-1), return_distance=False)
	#print(neighbours)
        val, vec = calcAccuracy(neighbours, i)
        avg = avg + val
	
    avg = avg/len(y)
    print(avg)
    #print(avg_n)
    #print(ndcg)

knn = NearestNeighbors(n_neighbors=11)
knn.fit(y)
NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=10, p=2, radius=1.0)
for j in range(2,11):
    avg_n = 0.0
    for i in range(0, len(y)):
        neighbours = knn.kneighbors(np.array(y[i]).reshape(1,-1), return_distance=False)
	#print(neighbours)
        val, vec = calcAccuracy(neighbours, i)
	#print vec
	#print j
        #print ndcg_at_k(vec, j-1)
        avg_n = avg_n + ndcg_at_k(vec, j-1)
    avg_n = avg_n/len(y)
    print(avg_n)


