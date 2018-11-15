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
import pickle
from nltk.tag import StanfordNERTagger
import random

File = open("questions") #open file
data = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

jar = 'stanford-ner.jar'
model = 'english.all.3class.distsim.crf.ser.gz'

st = StanfordNERTagger(model, jar, encoding='utf8') 

#tokenized_sents = [[nltk.word_tokenize(str(sent)) for sent in nltk.sent_tokenize(str(line))] for line in lines]
#classified_text = st.tag_sents(tokenized_sents)

tokenized_sents = []
persons = []
entities = []
        
for line in data:
    sentences = nltk.sent_tokenize(line) #tokenize sentences
    tokenized_sents.append(nltk.word_tokenize(str(sentences)))
    for sentence in sentences:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NNS'):
		if (word not in entities):
	            entities.append(word)

classified_text = st.tag_sents(tokenized_sents)

for item in classified_text:
    for x,y in item:
        if (y == 'PERSON'):
            if (x not in persons):
	       persons.append(str(x))

def create_data(x):
    tr_x = []
    all_problems = []
    tr_y = []
    n = len(x)
    for i in range(0, n):
        tr_x += [x[i]]
        all_problems.append(x[i])
        tr_y += [x[i]]
        #NP transform
        sentences = nltk.sent_tokenize(x[i]) #tokenize sentences
    	curr_persons = []
        curr_entities = []
        for a,y in classified_text[i]:
            if (y == 'PERSON'):
                if (a not in curr_persons):
                    curr_persons.append(str(a))
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentences))):
            if (pos == 'NNS'):
		if (word not in curr_entities and word in entities):
	            curr_entities.append(word)
        num_persons = len(curr_persons)
        num_entities = len(curr_entities)
        new_prob = x[i]
        for j in range(0, num_persons):
	    repl_person = persons[random.randint(0, len(persons)-1)]
            new_prob = new_prob.replace(curr_persons[j], repl_person)
        if (num_persons != 0):
            all_problems.append(new_prob)
            tr_x += [x[i]]
            tr_y += [new_prob]
        new_prob = x[i]
        new_prob1 = x[i]
        for j in range(0, num_entities):
	    repl_entity = entities[random.randint(0, len(entities)-1)]
            new_prob1 = new_prob1.replace(curr_entities[j], repl_entity)
            new_prob = new_prob.replace(curr_entities[j], repl_entity)
        if (new_prob not in all_problems):
            all_problems.append(new_prob)
            tr_x += [x[i]]
            tr_y += [new_prob]
        if (new_prob1 not in all_problems):
            all_problems.append(new_prob1)
            tr_x += [x[i]]
            tr_y += [new_prob]

    return tr_x, tr_y, all_problems


tr_x, tr_y, all_problems = create_data(data)
tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(all_problems)
#print(tokenizer.word_index)  # To see the dicstionary

TR_X = tokenizer.texts_to_sequences(tr_x)
TR_X = pad_sequences(TR_X, maxlen = 40)

TR_Y = tokenizer.texts_to_sequences(tr_y)
TR_Y = pad_sequences(TR_Y, maxlen = 40)

encode_data = tokenizer.texts_to_sequences(data)
encode_data = pad_sequences(encode_data, maxlen = 40)

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
                            input_length=40,
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
encoding_dim = 300

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



embed_dim = 128
lstm_out = 200
batch_size = 32

model = Sequential()
model.add(embedding_layer)
#model.add(LSTM(lstm_out, return_sequences = True, dropout_U = 0.2, dropout_W = 0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(50*40, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='sgd')
print(model.summary())


embedding_model = Model(inputs=model.input,
                                 outputs=model.layers[1].output)
embedded_wp = embedding_model.predict(TR_Y)

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
embedding_model = Model(inputs=model.input,
                                 outputs=model.layers[1].output)
embedded_wp = embedding_model.predict(TR_Y)

model.fit(TR_X, embedded_wp,
          epochs=500,
          batch_size=256,
          shuffle=True)
          #validation_data=(x_test, x_test))


#encoded_wp = encoder.predict(X)

print(model.summary())



intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[4].output)
encoded_wp = intermediate_layer_model.predict(encode_data)


avg = 0.0
arrY = np.asarray(encoded_wp)

print(arrY.shape)

outfile = open("autoenc_model_perturb","wb")
pickle.dump(arrY, outfile)
outfile.close()

