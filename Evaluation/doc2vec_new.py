from gensim import models
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import pickle
import random

File = open("questions") #open file
questions = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in questions]
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

indices = []

for i in range(0, len(data)):
    indices.append(i)

train_indices = random.sample(indices, 450)
test_indices = []

train_labels = []
train_data = [] 
test_data = []
test_labels = []

for i in range(0, len(data)):
    if (i in train_indices):
	train_data.append(data[i])
	train_labels.append(docLabels[i])
    else:
	test_data.append(data[i])
	test_labels.append(docLabels[i])
	test_indices.append(i)

x_train = LabeledLineSentence(train_data, train_labels)
x_test = test_data

print(len(train_data))
print(len(test_data))

it = LabeledLineSentence(data, docLabels)

model = gensim.models.Doc2Vec(min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
for epoch in range(10):
 print 'iteration'+str(epoch+1)
 model.train(x_train, total_examples = len(x_train.doc_list), epochs = 10)
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
for i in range(0, len(x_test)):
    y.append(model.infer_vector(x_test[i]))


outfile = open("retrieval_indices","w")
for indice in test_indices:
    outfile.write(str(indice)+"\n")
outfile.close()

outfile = open("doc2vec_model","wb")
pickle.dump(y, outfile)
outfile.close()

