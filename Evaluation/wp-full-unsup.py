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
import nltk
from nltk.tag import StanfordNERTagger


import random
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Embedding, Bidirectional, LSTM
from keras.optimizers import RMSprop
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn import cross_validation
import pickle

num_classes = 4
epochs = 100

File = open("questions") #open file
rawdata = File.readlines() #read all lines
File.close()

File = open("schemas-cl") #open file
schemas = File.readlines()
File.close()

schemas = map(int, schemas)

# you may also want to remove whitespace characters like `\n` at the end of each line
data = [x.strip() for x in rawdata]

print(len(data))

entities = []
persons = []
verbs = []

lines = rawdata
jar = 'stanford-ner.jar'
model = 'english.all.3class.distsim.crf.ser.gz'

st = StanfordNERTagger(model, jar, encoding='utf8') 

#tokenized_sents = [[nltk.word_tokenize(str(sent)) for sent in nltk.sent_tokenize(str(line))] for line in lines]
#classified_text = st.tag_sents(tokenized_sents)

tokenized_sents = []
        
for line in lines:
    sentences = nltk.sent_tokenize(line) #tokenize sentences
    tokenized_sents.append(nltk.word_tokenize(str(sentences)))
    for sentence in sentences:
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
            if (pos == 'NNS'):
		if (word not in entities):
	            entities.append(word)
            if ('VB' in pos):
		if (word not in verbs):
	            verbs.append(word)

classified_text = st.tag_sents(tokenized_sents)

for item in classified_text:
    for x,y in item:
        if (y == 'PERSON'):
            if (x not in persons):
	       persons.append(str(x))

print(entities)
print(len(entities))
print(persons)
print(len(persons))

outfile = open("persons","wb")
pickle.dump(entities, outfile)
outfile.close()

outfile = open("entities","wb")
pickle.dump(persons, outfile)
outfile.close()

def create_pairs(x, persons, entities):
    pairs = []
    labels = []
    all_problems = []
    n = len(x)
    print(x[1])
    #same problem
    for i in range(0, n):
        print(i)
        all_problems.append(x[i])
        pairs += [[x[i], x[i]]]
        labels += [1]
        #continue
        #NP transform
        sentences = nltk.sent_tokenize(x[i]) #tokenize sentences
    	curr_persons = []
        curr_entities = []
        curr_verbs = []
        for a,y in classified_text[i]:
            if (y == 'PERSON'):
                if (a not in curr_persons):
                    curr_persons.append(str(a))
        for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentences))):
            if (pos == 'NNS'):
		if (word not in curr_entities and word in entities):
	            curr_entities.append(word)
            if ('VB' in pos):
		if (word not in curr_verbs and word in verbs):
	            curr_verbs.append(word)

        num_persons = len(curr_persons)
        num_entities = len(curr_entities)
        num_verbs = len(curr_verbs)
        print(curr_persons)
        print(curr_entities)
        print(curr_verbs)
        new_prob = x[i]
        for j in range(0, num_persons):
	    repl_person = persons[random.randint(0, len(persons)-1)]
            new_prob = new_prob.replace(curr_persons[j], repl_person)
        if (num_persons != 0):
            all_problems.append(new_prob)
            pairs += [[x[i], new_prob]]
            labels += [1]
        new_prob = x[i]
        new_prob1 = x[i]
        for j in range(0, num_entities):
	    repl_entity = entities[random.randint(0, len(entities)-1)]
            new_prob1 = new_prob1.replace(curr_entities[j], repl_entity)
            new_prob = new_prob.replace(curr_entities[j], repl_entity)
        if (new_prob not in all_problems):
            all_problems.append(new_prob)
            pairs += [[x[i], new_prob]]
            labels += [1]
        if (new_prob1 not in all_problems):
            all_problems.append(new_prob1)
            pairs += [[x[i], new_prob1]]
            labels += [1]

        new_prob2 = x[i]
        for j in range(0, num_verbs):
	    repl_verb = verbs[random.randint(0, len(verbs)-1)]
            new_prob2 = new_prob2.replace(curr_verbs[j], repl_verb)
	    if (new_prob2 not in all_problems):
                all_problems.append(new_prob2)
                pairs += [[x[i], new_prob2]]
                labels += [0]

    return pairs, np.array(labels), all_problems


pairs, labels, all_problems = create_pairs(data, persons, entities)

outfile = open("all_problems","w")
for i in range(0, len(all_problems)):
    outfile.write(all_problems[i]+"\n")
outfile.close()

'''
#schema templates

change_template = "(person1) has (num1) (entity1). (person1) added (num2) (entity1). How many (entity1) does (person1) have?"
combine_template = "(person1) has (num1) (entity1). (person2) has (num2) (entity1). How many (entity1) are there altogether?"
vary_template = "(person1) has (num1) (entity1). (person2) has (num2) times as many (entity1). How many (entity1) does (person2) have?"
compare_template = "(person1) has (num1) (entity1). (person2) has (num2) (entity1) more than (person1). How many (entity1) does (person2) have?"

#fill template

def get_filled_template(problem, curr_persons, curr_entites, curr_num):
    schema = ""
    if (("each" in problem) or ("every" in problem) or ("per" in problem) or ("equally" in problem)):
	schema = "vary"
    elif (" than" in problem):
        schema = "compare"
    elif (("total" in problem) or ("together" in problem) or ("in all" in problem) or ("altogether" in problem)):
	schema = "combine"
    else
        schema = "change"
    schemas = ["vary", "compare", "combine", "change"]
    new_problems = []
    start = schemas.index(schema)
    person1 = "", person2 = "", entity = "", number1 = "", number2 = ""
    if (len(curr_persons) >= 2):
	person1 = curr_persons[0]
	person2 = curr_persons[1]
    elif (len(curr
'''


tokenizer = Tokenizer(nb_words=100, lower=True,split=' ')
tokenizer.fit_on_texts(all_problems)
#print(tokenizer.word_index)  # To see the dictionary
X = tokenizer.texts_to_sequences(all_problems)
X = pad_sequences(X)

print(X.shape)

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
                            input_length=X.shape[1],
                            trainable=False)

model1 = Sequential()
model1.add(embedding_layer)
model1.compile('rmsprop', 'mse')
ans = model1.predict(X)

X1 = tokenizer.texts_to_sequences(data)
X1 = pad_sequences(X1, maxlen = X.shape[1])
ans1 = model1.predict(X1)

print(ans.shape)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_vec_pairs(pairs, labels, all_problems, ans):
    n = len(pairs)
    vec_pairs = []
    for i in range(n):
	pair1 = ans[all_problems.index(pairs[i][0])]
	pair2 = ans[all_problems.index(pairs[i][1])]
        vec_pairs += [[pair1, pair2]]
    return np.array(vec_pairs), np.array(labels)

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Bidirectional(LSTM(64))(input)
    m = Model(input, x)
    print(m.summary())
    return m


def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

'''
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
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
te_pairs, te_y = create_pairs(x_test, digit_indices)
'''

tr_pairs, tr_y = create_vec_pairs(pairs, labels, all_problems, ans)
input_shape = ans.shape[1:]

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
          epochs=epochs)
#          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

print(model.summary())


input_c = Input(shape=input_shape)

print(model.layers[0].output_shape)
print(model.layers[1].output_shape)
print(model.layers[2].output_shape)

pred_model = Model(inputs=input_a,
                                 outputs=processed_a)
embedded_wp = pred_model.predict(ans1)

outfile = open("siamese_model_perturb","wb")
pickle.dump(embedded_wp, outfile)
outfile.close()

'''
# compute final accuracy on training and test sets
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
'''
