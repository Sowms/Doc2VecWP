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
#import tsne
#from tsne import bh_sne

'''
glove_file = 'glove.6B.50d.txt'
tmp_file = get_tmpfile("test_word2vec.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

#nltk.download('stopwords')

#dict_std = {"ENT0" : np.random.random(50), "CONT0" : np.random.random(50), "CONT1" : np.random.random(50)}

'''

'''
def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    vecs = []
    doc1 = [word for word in doc if word in word2vec_model.vocab]
    vecs = word2vec_model[doc1] 

    for word in doc:
        if (word in word2vec_model.vocab):
	    vecs.append(word2vec_model[word])
	if (word in dict_std):
	    vecs.append(dict_std[word])
    
    all_vecs = []
    for i in range(0, len(vecs)-3):
	vec = vecs[i] + vecs[i+1] + vecs[i+2]
	all_vecs.append(vec)

    return np.mean(vecs, axis=0)
'''

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    #doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc


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

File = open("remaining-illinois") #open file
data = File.readlines() #read all lines
File.close()

'''
with open('q2-num-ent') as f:
    ques = f.readlines()
f.close()
'''

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

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
#print data[0]


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


f1 = open("remaining-output","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))
    #print i.rstrip() + '|' + `l[counter]`;

'''
ques = nlp_clean(ques)

print(ques[0])
'''

y = []
for i in range(0,len(data)):
    y.append(model.docvecs[docLabels[i]])
    #y.append(document_vector(model, data[i]))

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


X_train, X_test, y_train, y_test = cross_validation.train_test_split(y, intoutput, test_size=0.2, random_state=17)
classifier = train_classifier(X_train,y_train)
print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
print (classifier.score(X_test,y_test))

'''
#testing sandbox
f1 = open("sandbox.txt","r")
testx = f1.readlines();
f1.close()

inptest = []
for i in range(0,len(testx)):
    inptest.append(model.infer_vector(testx[i]))

f1 = open("sandbox-output","r")
testy = f1.readlines();
f1.close()

outtest = []
for i in testy:
    i = i.rstrip();
    outtest.append(int(i))

print(len(inptest))
print(len(outtest))
classifier = train_classifier(y, intoutput)
print (classifier.score(inptest,outtest))
'''

avg = 0.0

#print(y[0][0])


for j in range(0, 100):
    arrY = np.asarray(y)
    kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)  
    X = kmeans_model.fit(arrY)
    labels = kmeans_model.labels_.tolist()
    l = kmeans_model.fit_predict(arrY)
    #computing purity
    #3 clusters in l and 3 classes in output 
    #Step 1 - identifying N
    N = len(intoutput)
    #Step 2 - aggregating each cluster
    matrix = [[0 for x in range(4)] for m in range(4)] 
    matrix = numpy.zeros_like(matrix)
    for i in range(0, len(labels)):
    #print(`labels[i]` + "|" + `intoutput[i]`)
        matrix[labels[i]][intoutput[i]] += 1
    num = 0
    for i in range(0, 4):
        num += max(matrix[i])
    purity = num/float(N)
    avg = avg + purity
avg = avg/100
print(avg)


'''
x_data = np.asarray(y).astype('float64')
#x_data = x_data.reshape((x_data.shape[0], -1))
print(x_data[0])
X_2d = bh_sne(x_data, perplexity = 5)
label1 = ["#FF0000", "#FFFF00", "#008000", "#0000FF"]
color = [label1[i] for i in intoutput]
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()

arrY = np.asarray(y)
kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)  
X = kmeans_model.fit(arrY)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(arrY)
label1 = ["#FF0000","#FFFF00", "#008000", "#0000FF"]
color = [label1[i] for i in l]
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()
'''

'''
pca = PCA(n_components=2).fit(y)
datapoint = pca.transform(y)

plt.figure
colors = ["#FFFF00", "#008000", "#0000FF","#FF0000", "#00FF00", "#00FFFF"]
color = [colors[i] for i in intoutput]
labels = ['wp{0}'.format(i) for i in range(0, len(y))]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
for label, x, y in zip(labels, datapoint[:, 0], datapoint[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-5, 5),
        textcoords='offset points', ha='right', va='bottom')
        #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        #arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.show()
'''

'''
distances = []
for i in xrange (0, len(intoutput), 2):
    print(i)
    distances.append(numpy.linalg.norm(y[i] - y[i+1]))
#print(distances)

plt.figure
xcoord = [(i+1) for i in range(0, len(intoutput)/2)]
plt.scatter(xcoord, distances)
plt.show()
'''


'''
arrY = np.asarray(y)
kmeans_model = KMeans(n_clusters=4, init='k-means++', max_iter=100)  
X = kmeans_model.fit(arrY)
labels = kmeans_model.labels_.tolist()
l = kmeans_model.fit_predict(arrY)
color = [label1[i] for i in l]    
plt.figure
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
plt.show()
'''
