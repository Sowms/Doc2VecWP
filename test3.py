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

nltk.download('stopwords')

with open('q1.txt') as f:
    data = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]
print data[0]

docLabels = []
for counter in range(0, len(data)):
    docLabels.append('wp' + `counter`)
print docLabels[0]

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

kmeans_model = KMeans(n_clusters=3, init='k-means++', max_iter=100)  
X = kmeans_model.fit(model.docvecs.doctag_syn0)
labels = kmeans_model.labels_.tolist()

l = kmeans_model.fit_predict(model.docvecs.doctag_syn0)
f1 = open("q1-schema","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))
    #print i.rstrip() + '|' + `l[counter]`;
print(intoutput)
pca = PCA(n_components=2).fit(model.docvecs.doctag_syn0)
datapoint = pca.transform(model.docvecs.doctag_syn0)

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
#plt.show()

plt.figure
label1 = ["#FFFF00", "#008000", "#0000FF","#FFFF00"]
color = [label1[i] for i in intoutput]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)
#plt.show()


#computing purity
#3 clusters in l and 3 classes in output 
#Step 1 - identifying N
N = len(output)
#Step 2 - aggregating each cluster
matrix = [[0 for x in range(3)] for y in range(3)] 
print(len(matrix[0]))
matrix = numpy.zeros_like(matrix)
for i in range(0, len(labels)):
    #print(`labels[i]` + "|" + `intoutput[i]`)
    matrix[labels[i]][intoutput[i]-1] += 1
print(matrix)
num = 0
for i in range(0, 3):
    num += max(matrix[i])
purity = num/float(N)
print purity    


