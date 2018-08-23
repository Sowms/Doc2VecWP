#https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Embedding, LSTM, Flatten
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from sklearn import cross_validation
from sklearn.cluster import KMeans

File = open("complete-illinois-questions") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)  # To see the dicstionary
X = tokenizer.texts_to_sequences(data)
X = pad_sequences(X)
word_index = tokenizer.word_index

embeddings_index = {}
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            50,
                            weights=[embedding_matrix],
                            input_length=30,
                            trainable=False)


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

# this is the size of our encoded representations
encoding_dim = 32  

'''
# this is our input placeholder
input_wp = Input(shape=(200,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_wp)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(200, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
#autoencoder = Model(input_wp, decoded)
'''

intoutput = [0] * len(data)

X = np.asarray(X)
X = X.astype('float32')

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, intoutput, test_size=0.2, random_state=17)


embed_dim = 128
lstm_out = 200
batch_size = 32

print(X.shape[1])

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(lstm_out, return_sequences = True, dropout_U = 0.2, dropout_W = 0.2))
model.add(Flatten())
model.add(Dense(encoding_dim, activation='relu'))
model.add(Dense(30*50, activation='sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[3].output)
encoded_wp = intermediate_layer_model.predict(X)

embedding_model = Model(inputs=model.input,
                                 outputs=model.layers[0].output)
embedded_wp = embedding_model.predict(X)

arrY = np.asarray(embedded_wp)
arrY = arrY.reshape(len(arrY), 30*50)


'''
input_wp = Input(shape=(200,))
# this model maps an input to its encoded representation
encoder = Model(input_wp, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = model.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
'''

model.fit(X, arrY,
          epochs=50,
          batch_size=256,
          shuffle=True)
          #validation_data=(x_test, x_test))

#encoded_wp = encoder.predict(X)

f1 = open("illinois-output","r")
output = f1.readlines();
f1.close()

intoutput = []
for i in output:
    i = i.rstrip();
    intoutput.append(int(i))


avg = 0.0
arrY = np.asarray(encoded_wp)
#arrY = arrY.reshape(len(arrY), 30*32)

for j in range(0, 100):
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



