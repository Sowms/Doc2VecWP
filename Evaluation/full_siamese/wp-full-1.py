'''Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
https://github.com/keras-team/keras/blob/master/examples/mnist_siamese.py
'''
from __future__ import absolute_import
from __future__ import print_function
import numpy as np

import random
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding, Bidirectional, LSTM
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
import pickle
from sklearn.neighbors import NearestNeighbors
from ndcg import ndcg_at_k

num_classes = 4
epochs = 20

File = open("questions") #open file
data = File.readlines() #read all lines
File.close()

File = open("schemas-cl") #open file
schemas = File.readlines()
File.close()

schemas = map(int, schemas)

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in data]

print(len(data))

tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(data)
#print(tokenizer.word_index)  # To see the dictionary
X = tokenizer.texts_to_sequences(data)
X = pad_sequences(X)

print(X.shape)

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
                            input_length=30,
                            trainable=False)

model1 = Sequential()
model1.add(embedding_layer)
model1.compile('rmsprop', 'mse')
ans = model1.predict(X)

print(ans.shape)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, out):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = len(x)
    for i in range(n):
	for j in range(i+1, n-1):
	    pairs += [[x[i], x[j]]]
            if (out[i] == out[j]):
		labels += [1]
	    else:
		labels+=[0]
    return np.array(pairs), np.array(labels)


def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    #x = Flatten()(input)
    x = Bidirectional(LSTM(64))(input)
    '''
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    '''
    m = Model(input, x)
    print(m.summary())
    return m


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


# the data, split between train and test sets

x_train, x_test, y_train, y_test = cross_validation.train_test_split(ans, schemas, test_size=0.2, random_state=17)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train= np.asarray(y_train)
y_test= np.asarray(y_test)
input_shape = x_train.shape[1:]

print(x_train.shape)
print(input_shape)
print(y_train)
print(type(y_train[0]))

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(x_train, y_train)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, y_test)

print(tr_pairs.shape)
print(te_pairs.shape)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance,
                  output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

# train
rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          batch_size=128,
          epochs=epochs,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

print(model.summary())


input_c = Input(shape=input_shape)

print(model.layers[0].output_shape)
print(model.layers[1].output_shape)
print(model.layers[2].output_shape)

pred_model = Model(inputs=input_a,
                                 outputs=processed_a)
embedded_wp = pred_model.predict(ans)

outfile = open("siamese_model","wb")
pickle.dump(embedded_wp, outfile)
outfile.close()

#model.save('full-model')
#pred_model.save('infer-model')

# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

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


y = embedded_wp
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
