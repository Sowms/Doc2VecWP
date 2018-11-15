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
import random

glove_file = 'glove.6B.50d.txt'
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

indices = []

for i in range(0, len(data)):
    indices.append(i)
    
train_indices = random.sample(indices, 450)
test_indices = []
train_data = []
test_data = []
test_output = []

for i in range(0, len(data)):
    if (i in train_indices):
	train_data.append(data[i])
    else:
	test_data.append(data[i])
	test_indices.append(i)
	test_output.append(y[i])

outfile = open("retrieval_indices","w")
for indice in test_indices:
    outfile.write(str(indice)+"\n")
outfile.close()

outfile = open("avgword2vec_model","wb")
pickle.dump(test_output, outfile)
outfile.close()
