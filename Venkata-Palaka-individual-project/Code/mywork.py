#Importing Libraries
import numpy as np
from tkinter import *
from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
import json
import string
from utilities import *
import random
import nltk
from nltk.corpus import stopwords

max_sentence_length = 10
word_to_index, index_to_word, word_to_vector = read_glove_vecs('glove.6B.50d.txt')


def convert_sent_indices(X, word_to_index, max_len):

    char = {key: None for key in string.punctuation}
    char['"'] = None
    table = str.maketrans(char)
    stop_words = set(stopwords.words('english'))

    m = X.shape[0]

    X_indices = np.zeros(shape=(m, max_len))

    for i in range(m):
        # Removing punctuation
        X[i] = X[i].translate(table)

        # Converting the training sentence into lower case and split into words.
        sentence_words = X[i].lower().split()

        j = 0

        for k in sentence_words:
            # Skip Stopwords
            if k in stop_words:
                continue

            if k in word_to_index:
                X_indices[i, j] = word_to_index[k]

            else:
                pass
            j += 1

    return X_indices

# Loading saved model
m = '/home/ubuntu/nlp/Chatbot_Glove_model/pretrained model/trained_lstm_128_128_dropout_4_3.h5'
model = load_model(m)

# Training data
X_train = []
Y_train = []

res = [] #responses
tags = [] #tags
tags_index = [] #index of all tags

with open('/home/ubuntu/nlp/Chatbot_Glove_model/data/intents.json') as json_data:
    data = json.load(json_data)
    index_counter = 0
    for i in data['intents']:
        tag = i['tag']
        pattern = i['patterns']
        response = i['responses']

        tag_index = index_counter
        index_counter += 1

        tags.append(tag)
        tags_index.append(tag_index)
        res.append(response)

        for j in i['patterns']:
            X_train.append(j)
            Y_train.append(tag_index)


d = {} #dictionary to store tags with corresponding responses
for key in tags:
    for value in res:
        d[key] = value
        res.remove(value)
        break

#saving all the tags in a text file
with open('tags.txt', 'w+') as f:
    for i in range(0, len(tags)):
        f.write('{}\t\t\t{}\n'.format(tags_index[i], tags[i]))
z = len(tags)

#print(X_train[:10])
#print(Y_train[:10])
#print(len(X_train))
#print(len(Y_train))

# Convert training data to numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Testing
test_sent = ['what are the documents required for admission through Minority Quota',
                 'what is the admission procedure for computer engineering',
                 'what is the admission procedure for mechanical engineering',
                 'what is the admission procedure for electrical engineering',
                 'hi there',
                 'thanks for your help',
                 'bye, thanks',
                 'what documents are required for obc',
                 'what is the timing for office',
                 'How can I apply as NRI?',
                 'What will be the charge for application form?',
                 'what time is it open today']

X_test = convert_sent_indices(np.array(test_sent), word_to_index, max_sentence_length)
prediction = model.predict(X_test)
prediction_index = np.argmax(prediction, axis=1)

for i in range(len(test_sent)):
    print(test_sent[i])
    print(str(prediction_index[i]) + '   Expected Intent :  ' + tags[prediction_index[i]] + '\n')

model.summary()

max_length = len(max(X_train, key=len).split())
#print(max_length)

Y_Train = convert_to_one_hot(Y_train, C=z)

#print(Y_train[:6])
#print(Y_Train[:6])


def pretrained_embed_layer(word_to_vector, word_to_index):

    vocab_length = len(word_to_index) + 1  # adding 1 to fit Keras embedding
    embed_dimension = word_to_vector["cucumber"].shape[0]  # dimensionality of GloVe word vectors (= 50)

    # Initializing the embedding matrix as a numpy array of zeros
    embed_matrix = np.zeros(shape=(vocab_length, embed_dimension))

    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        embed_matrix[index, :] = word_to_vector[word]

    # Defining Keras embedding layer
    embedding_layer = Embedding(vocab_length, embed_dimension, trainable=False)

    # Build the embedding layer
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix.
    embedding_layer.set_weights([embed_matrix])

    return embedding_layer


def embed_func(input_shape, word_to_vector, word_to_index):


    # Defining sentence_indices as the input_shape and dtype 'int32' (as it contains indices).
    sentence_indices = Input(shape=input_shape)

    # Creating the embedding layer pretrained with GloVe Vectors
    embedding_layer = pretrained_embed_layer(word_to_vector, word_to_index)

    # Propagating sentence_indices through embedding layer
    embeddings = embedding_layer(sentence_indices)

    # Propagating the embeddings through an LSTM layer with 128-dimensional hidden state
    X = LSTM(128, return_sequences=True)(embeddings)
    # Adding dropout
    X = Dropout(0.4)(X)
    # Propagating X through another LSTM layer with 128-dimensional hidden state
    X = LSTM(128)(X)
    # Adding dropout
    X = Dropout(0.3)(X)
    # Propagating X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(z)(X)
    # Adding softmax activation
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model


model = embed_func((max_sentence_length,), word_to_vector, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_Train_indices = convert_sent_indices(X_train, word_to_index, max_sentence_length)

model.fit(X_Train_indices, Y_Train, epochs=100, batch_size=32, shuffle=True)

print(X_Train_indices)
print(Y_Train)
print(X_Train_indices.shape)
print(Y_Train.shape)

#Calculating Model's loss and Accuracy

loss, accuracy = model.evaluate(X_Train_indices, Y_Train)
#print("loss:",loss,"Accuracy:",accuracy)

"""
Testing on a single sentence
test_sent = ['Where do I find the link for faculty information']
print("Sample Question:", test_sent[0])

X_test = convert_sent_indices(np.array(test_sent), word_to_index, max_sentence_length)
pred = model.predict(X_test)
pred_index = np.argmax(pred)
# print(pred_index)
# print(tags[pred_index])
s = tags[pred_index]
# print(d[s]) #all the responses
print("Reply:", random.choice(d[s]))
"""

#saving model
model.save('trained_lstm.h5')

#integrating lstm model to chatbot
def botResponse(sentence):
    sent = [sentence]
    X_test = convert_sent_indices(np.array(sent), word_to_index, max_sentence_length)
    prediction = model.predict(X_test)
    prediction_index = np.argmax(prediction)

    if tags[prediction_index] not in d.keys():
        b = "sorry, I can't understand!"
        return b
    else:
        s = tags[prediction_index]
        return random.choice(d[s])



"""
Utilities.py 
import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename = 'data/emojify_data.csv'):
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

"""