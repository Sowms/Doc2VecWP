from gensim import models
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim

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

