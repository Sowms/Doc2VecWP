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
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import json
import numpy as np
from pprint import pprint

nltk.download('stopwords')
'''
with open('AddSub.json') as f1:
    dataAddSub = json.load(f1)
f1.close()

data = []

for i in range(0, len(dataAddSub)):
    data.append(dataAddSub[i]["sQuestion"])

with open('SingleOp.json') as f2:
    dataSingleOp = json.load(f2)
f2.close()

for i in range(0, len(dataSingleOp)):
    data.append(dataSingleOp[i]["sQuestion"])
'''

File = open("entire-num-prop-ent") #open file
data = File.readlines() #read all lines
File.close()


with open('q2.txt') as f:
    ques = f.readlines()
f.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]
print data[0]

docLabels = []
for counter in range(0, len(data)):
    docLabels.append('wp' + `counter`)
#print docLabels[0]

#https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104

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

data = nlp_clean(data)
print data[0]
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
print "model saved"

#f1 = open("vec.txt","w+")
#for counter in range(0, len(data)):
#	f1.write(model.docvecs[docLabels[counter]])
#f1.close()

#https://www.kaggle.com/sgunjan05/document-clustering-using-doc2vec-word2vec

f1 = open("q2-schema","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))
    #print i.rstrip() + '|' + `l[counter]`;
print(intoutput)

ques = nlp_clean(ques)

print(ques[0])

y = []
for i in range(0,len(ques)):
    y.append(model.infer_vector(ques[i], alpha = 0.01, min_alpha = 0.001, steps = 5))

print(len(y))

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

X_train, X_test, y_train, y_test = cross_validation.train_test_split(y, intoutput, test_size=0.02, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))

avg = 0.0

print(y[0][0])

for j in range(0, 100):
    arrY = np.asarray(y)
    kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=100)  
    X = kmeans_model.fit(arrY)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(arrY)
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
    purity = num/float(N)
    avg = avg + purity
avg = avg/100
print(avg)
