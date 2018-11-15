#https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
import numpy as np
from gensim import models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
import math
import pickle
import random
from sklearn import cross_validation
from sklearn.neighbors import NearestNeighbors
from ndcg import ndcg_at_k
from numpy.random import seed
seed(27)
from tensorflow import set_random_seed
set_random_seed(2)

File = open("questions") #open file
data = File.readlines() #read all lines
File.close()

File = open("schemas") #open file
schemas = File.readlines() #read all lines
File.close()

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]


tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)  # To see the dictionary
X = tokenizer.texts_to_sequences(data)
X = pad_sequences(X, maxlen = 31)

print(X.shape)

X1 = X[:, 1:]
X1 = np.pad(X1,(0,1),'constant')
X1 = np.delete(X1, (len(X)), axis=0)
print(X[0])
print(X1.shape)

word_index = tokenizer.word_index

embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=31,
                            trainable=False)

model1 = Sequential()
model1.add(embedding_layer)
model1.compile('rmsprop', 'mse')
ans = model1.predict(X)
ans1 = model1.predict(X1)
print(ans.shape)

X2 = tokenizer.texts_to_sequences(data)
X2 = pad_sequences(X2, maxlen=31)
ans2 = model1.predict(X2)

noise_factor = 0.5

#encoder_input_data = ans + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 
#decoder_input_data = ans + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 
#decoder_target_data = ans1 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 

x_noisy = ans + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 


encoder_input_data = np.concatenate((ans,x_noisy))
decoder_input_data = encoder_input_data
decoder_target_data = np.concatenate((ans1,ans1))


# configure
num_encoder_tokens = 100
num_decoder_tokens = 100
latent_dim = 256

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# plot the model
plot_model(model, to_file='model.png', show_shapes=True)

#encoder_input_data = x_train
#decoder_input_data = x_train
#decoder_target_data = y_train

#train
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=200)
#          validation_split=0.2)

# define encoder inference model
encoder_model = Model(encoder_inputs, encoder_states)
encoded_rep = np.asarray(encoder_model.predict(ans2))
encoded_arr = encoded_rep[0]
# define decoder inference model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
# summarize model
plot_model(encoder_model, to_file='encoder_model.png', show_shapes=True)
plot_model(decoder_model, to_file='decoder_model.png', show_shapes=True)

avg = 0.0
arrY = encoded_arr


outfile = open("autoenc_model","wb")
pickle.dump(arrY, outfile)
outfile.close()

y = arrY


def CalcNDCG(Vector, k):
    dScore = 0.0;
    length = k
    SortedVector = []
    DCGVector = []
    DCGSortedVector = []
    for i in range(0, length):
        SortedVector.append(Vector[i])
    for i in range(0, length):
	for j in range(0, length):
            if(SortedVector[i] < SortedVector[j]):
		temp = SortedVector[j];
		SortedVector[j] = SortedVector[i];
		SortedVector[i] = temp; 
		
    DCGVector.append(Vector[0])
    DCGSortedVector.append(SortedVector[0])
    if (DCGSortedVector[0]==0):
        dScore=0.0
    else:
        dScore += float(DCGVector[0])/DCGSortedVector[0]
    
    for i in range (1, length):
        DCGVector.append(DCGVector[i-1] + (Vector[i]/(math.log(i+1)/math.log(2.0))))
	DCGSortedVector.append(DCGSortedVector[i-1] + (SortedVector[i]/(math.log(i+1)/math.log(2.0))))
	if (DCGSortedVector[i] != 0):
	    dScore += float(DCGVector[i])/DCGSortedVector[i]
        else:
	    dScore = 0
			
    #print (dScore)	
    return (dScore/length);


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
NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=10, p=2, radius=1.0)
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




    
 
