from gensim import models
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import pickle
from sklearn.neighbors import NearestNeighbors
from ndcg import ndcg_at_k
from numpy.random import seed
seed(27)
from tensorflow import set_random_seed
set_random_seed(2)
import numpy as np

File = open("questions") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]
#print data[0]

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

y = []
for i in range(0,len(data)):
    y.append(model.docvecs[docLabels[i]])

outfile = open("doc2vec_model","wb")
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
NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=11, p=2, radius=1.0)
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


