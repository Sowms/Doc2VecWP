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
import pickle
import random
from sklearn import cross_validation
from sklearn.neighbors import NearestNeighbors
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

indices = []

for i in range(0, len(data)):
    indices.append(i)
    
train_indices = random.sample(indices, 450)
test_indices = []
train_data = []
test_data = []
test_output = []

for i in range(0, len(data)):
    if (i in train_indices):
	train_data.append(data[i])
    else:
	test_data.append(data[i])
	test_indices.append(i)


tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)  # To see the dictionary
X = tokenizer.texts_to_sequences(train_data)
X = pad_sequences(X, maxlen = 31)

print(X.shape)

X1 = X[:, 1:]
X1 = np.pad(X1,(0,1),'constant')
X1 = np.delete(X1, (len(X)), axis=0)
print(X[0])
print(X1.shape)

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
                            input_length=31,
                            trainable=False)

model1 = Sequential()
model1.add(embedding_layer)
model1.compile('rmsprop', 'mse')
ans = model1.predict(X)
ans1 = model1.predict(X1)
print(ans.shape)

X2 = tokenizer.texts_to_sequences(test_data)
X2 = pad_sequences(X2, maxlen=31)
ans2 = model1.predict(X2)

noise_factor = 0.5
x1_train_noisy = ans + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 
x2_train_noisy = ans1 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans.shape) 
x_test_noisy = ans2 + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=ans2.shape) 

x1_train_noisy = np.clip(x1_train_noisy, 0., 1.)
x2_train_noisy = np.clip(x2_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


x_train = np.concatenate((ans,x1_train_noisy))
y_train = np.concatenate((ans,ans))

# configure
num_encoder_tokens = 50
num_decoder_tokens = 50
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

encoder_input_data = x_train
decoder_input_data = x_train
decoder_target_data = y_train

#train
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=200,
          validation_split=0.2)

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
r_indices = test_indices

def calcAccuracy(neighbours, wpIn):
    k = len(neighbours[0])
    schema = schemas[int(r_indices[wpIn])]
    match = 0.0
    for i in range(1, k):
	if (schemas[int(r_indices[neighbours[0][i]])] == schema):
	    match += 1
    return match/(k-1)


for j in range(2,11):
    knn = NearestNeighbors(n_neighbors=j)
    knn.fit(y)
    NearestNeighbors(algorithm='auto', leaf_size=30, n_neighbors=j, p=2, radius=1.0)
    avg = 0.0
    for i in range(0, len(y)):
        neighbours = knn.kneighbors(np.array(y[i]).reshape(1,-1), return_distance=False)
        val = calcAccuracy(neighbours, i)
        avg = avg + val
    avg = avg/len(y)
    print(avg)


