#https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from sklearn import cross_validation
from sklearn.cluster import KMeans

# this is the size of our encoded representations
encoding_dim = 32  

# this is our input placeholder
input_wp = Input(shape=(100,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_wp)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(100, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_wp, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_wp, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

File = open("complete-illinois-questions") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

docLabels = []
for counter in range(0, len(data)):
    docLabels.append('wp' + `counter`)

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

y = []
for i in range(0,len(data)):
    y.append(model.docvecs[docLabels[i]])

intoutput = [0] * len(y)

arrY = np.asarray(y)
arrY = arrY.astype('float32')

x_train, x_test, y_train, y_test = cross_validation.train_test_split(y, intoutput, test_size=0.2, random_state=17)

autoencoder.fit(arrY, arrY,
                epochs=50,
                batch_size=256,
                shuffle=True)
                #validation_data=(x_test, x_test))

encoded_wp = encoder.predict(arrY)

f1 = open("illinois-output","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))


avg = 0.0

for j in range(0, 100):
    arrY = np.asarray(encoded_wp)
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
    matrix = np.zeros_like(matrix)
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



