# -*- coding: utf-8 -*-
"""assignment08.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yB9mCPB_mkdN8FSzipcYyJQRFEMFA5a7

# Setup

## Steps I took to get this working

1. Download files:
    * https://canvas.northwestern.edu/courses/101722/files/7323167/download?wrap=1

2. Run `run-chakin-to-get-embeddings-v001.py` locally

3. Upload the zip file to your drive (Same folder as last week.)

4. Upload `movie-reviews-negative` and `movie-reviews-positive` folder to your drive.

## Import Packages
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, LSTMCell, GRU, GRUCell, RNN, SimpleRNN, SimpleRNNCell, RNN, Dropout, Dense
from tensorflow.keras import Sequential
from pprint import PrettyPrinter as pp

printer = pp(depth=2, width=10)

"""## Install chakin

* https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb
"""

# Commented out IPython magic to ensure Python compatibility.
# Run if you haven't installed this yet
# Otherwise, comment this cell out
import pip

#!pip install chakin
# %pip install chakin

"""## Import More Libraries"""

import os
from collections import defaultdict
import zipfile

import chakin

"""## Mount Drive"""

# Remember to use your NU account or be consistent with where you want your files to be!

from google.colab import drive

drive.mount('/content/drive/')

"""## Starter Code - Embeddings & Sentiment

### Embeddings Start-Up Code
"""

# Gather embeddings via chakin
# Following methods described in
#    https://github.com/chakki-works/chakin

# As originally configured, this program downloads four
# pre-trained GloVe embeddings, saves them in a zip archive,
# and then unzips the archive to create the four word-to-embeddings
# text files for use in language models.

# Note that the downloading process can take about 10 minutes to complete.

chakin.search(lang='English')  # lists available indices in English

# Specify English embeddings file to download and install
# by index number, number of dimensions, and subfoder name
# Note that GloVe 50-, 100-, 200-, and 300-dimensional folders
# are downloaded with a single zip download
CHAKIN_INDEX = 11
NUMBER_OF_DIMENSIONS = 50
SUBFOLDER_NAME = "gloVe.6B"

#DATA_FOLDER = "embeddings"
DATA_FOLDER = '/content/drive/My Drive/422_data_files'
ZIP_FILE = os.path.join(DATA_FOLDER, "{}.zip".format(SUBFOLDER_NAME))
ZIP_FILE_ALT = "glove" + ZIP_FILE[5:]  # sometimes it's lowercase only...
UNZIP_FOLDER = os.path.join(DATA_FOLDER, SUBFOLDER_NAME)
if SUBFOLDER_NAME[-1] == "d":
    GLOVE_FILENAME = os.path.join(
        UNZIP_FOLDER, "{}.txt".format(SUBFOLDER_NAME))
else:
    GLOVE_FILENAME = os.path.join(UNZIP_FOLDER, "{}.{}d.txt".format(
        SUBFOLDER_NAME, NUMBER_OF_DIMENSIONS))

print(ZIP_FILE)

# This will take a while to download at first.

if not os.path.exists(ZIP_FILE) and not os.path.exists(UNZIP_FOLDER):
    #GloVe by Stanford is licensed Apache 2.0:
        #https://github.com/stanfordnlp/GloVe/blob/master/LICENSE
        #http://nlp.stanford.edu/data/glove.twitter.27B.zip
        #Copyright 2014 The Board of Trustees of The Leland Stanford Junior University
   print("Downloading embeddings to '{}'".format(ZIP_FILE))
   chakin.download(number=CHAKIN_INDEX, save_dir='.{}'.format(DATA_FOLDER))
else:
   print("Embeddings already downloaded.")

ZIP_FILE = '/content/drive/My Drive/422_data_files/glove.6B.zip'
with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
   print("Extracting embeddings to '{}'".format(UNZIP_FOLDER))
   zip_ref.extractall(UNZIP_FOLDER)

"""### Sentiment Start-Up Code"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re  # regular expressions
import nltk
from nltk.tokenize import TreebankWordTokenizer

RANDOM_SEED = 9999

# To make output stable across runs
def reset_graph(seed= RANDOM_SEED):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

REMOVE_STOPWORDS = False  # no stopword removal 

EVOCABSIZE = 10000  # specify desired size of pre-defined embedding vocabulary

# Select the pre-defined embeddings source        
# Define vocabulary size for the language model    
# Create a word_to_embedding_dict for GloVe.6B.50d
embeddings_directory = '/content/drive/My Drive/422_data_files/gloVe.6B'
filename = 'glove.6B.50d.txt'
embeddings_filename = os.path.join(embeddings_directory, filename)

# Utility function for loading embeddings follows methods described in
# https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer
# Creates the Python defaultdict dictionary word_to_embedding_dict
# for the requested pre-trained word embeddings
# 
# Note the use of defaultdict data structure from the Python Standard Library
# collections_defaultdict.py lets the caller specify a default value up front
# The default value will be retuned if the key is not a known dictionary key
# That is, unknown words are represented by a vector of zeros
# For word embeddings, this default value is a vector of zeros
# Documentation for the Python standard library:
#   Hellmann, D. 2017. The Python 3 Standard Library by Example. Boston: 
#     Addison-Wesley. [ISBN-13: 978-0-13-429105-5]
def load_embedding_from_disks(embeddings_filename, with_indexes=True):
    """
    Read a embeddings txt file. If `with_indexes=True`, 
    we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, 
    otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping 
    from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
  
    else:
        word_to_embedding_dict = dict()

    with open(embeddings_filename, 'r', encoding='utf-8') as embeddings_file:
        for (i, line) in enumerate(embeddings_file):

            split = line.split(' ')

            word = split[0]

            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )

            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    # Empty representation for unknown words.
    _WORD_NOT_FOUND = [0.0] * len(representation)
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(
            lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(
            index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict

print('\nLoading embeddings from', embeddings_filename)
word_to_index, index_to_embedding = \
    load_embedding_from_disks(embeddings_filename, with_indexes=True)
print("Embedding loaded from disks.")
# Note: unknown words have representations with values [0, 0, ..., 0]

# Additional background code from
# https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer
# shows the general structure of the data structures for word embeddings
# This code is modified for our purposes in language modeling 
vocab_size, embedding_dim = index_to_embedding.shape
print("Embedding is of shape: {}".format(index_to_embedding.shape))
print("This means (number of words, number of dimensions per word)\n")
print("The first words are words that tend occur more often.")

print("Note: for unknown words, the representation is an empty vector,\n"
      "and the index is the last one. The dictionnary has a limit:")
print("    {} --> {} --> {}".format("A word", "Index in embedding", 
      "Representation"))
word = "worsdfkljsdf"  # a word obviously not in the vocabulary
idx = word_to_index[word] # index for word obviously not in the vocabulary
complete_vocabulary_size = idx 
embd = list(np.array(index_to_embedding[idx], dtype=int)) # "int" compact print
print("    {} --> {} --> {}".format(word, idx, embd))
word = "the"
idx = word_to_index[word]
embd = list(index_to_embedding[idx])  # "int" for compact print only.
print("    {} --> {} --> {}".format(word, idx, embd))

# Show how to use embeddings dictionaries with a test sentence
# This is a famous typing exercise with all letters of the alphabet
# https://en.wikipedia.org/wiki/The_quick_brown_fox_jumps_over_the_lazy_dog
a_typing_test_sentence = 'The quick brown fox jumps over the lazy dog'
print('\nTest sentence: ', a_typing_test_sentence, '\n')
words_in_test_sentence = a_typing_test_sentence.split()

print('Test sentence embeddings from complete vocabulary of', 
      complete_vocabulary_size, 'words:\n')
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = index_to_embedding[word_to_index[word_]]
    print(word_ + ": ", embedding)

# Define vocabulary size for the language model    
# To reduce the size of the vocabulary to the n most frequently used words

def default_factory():
    return EVOCABSIZE  # last/unknown-word row in limited_index_to_embedding
# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index = defaultdict(default_factory, \
    {k: v for k, v in word_to_index.items() if v < EVOCABSIZE})

# Select the first EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding = index_to_embedding[0:EVOCABSIZE,:]
# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding = np.append(limited_index_to_embedding, 
    index_to_embedding[index_to_embedding.shape[0] - 1, :].\
        reshape(1,embedding_dim), 
    axis = 0)

# Verify the new vocabulary: should get same embeddings for test sentence
# Note that a small EVOCABSIZE may yield some zero vectors for embeddings
print('\nTest sentence embeddings from vocabulary of', EVOCABSIZE, 'words:\n')
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = limited_index_to_embedding[limited_word_to_index[word_]]
    print(word_ + ": ", embedding)

# Utility function to get file names within a directory
def listdir_no_hidden(path):
    start_list = os.listdir(path)
    end_list = []
    for file in start_list:
        if (not file.startswith('.')):
            end_list.append(file)
    return(end_list)

# define list of codes to be dropped from document
# carriage-returns, line-feeds, tabs
codelist = ['\r', '\n', '\t']   

# We will not remove stopwords in this exercise because they are
# important to keeping sentences intact
if REMOVE_STOPWORDS:
    print(nltk.corpus.stopwords.words('english'))

# previous analysis of a list of top terms showed a number of words, along 
# with contractions and other word strings to drop from further analysis, add
# these to the usual English stopwords to be dropped from a document collection
    more_stop_words = ['cant','didnt','doesnt','dont','goes','isnt','hes',\
        'shes','thats','theres','theyre','wont','youll','youre','youve', 'br'\
        've', 're', 'vs'] 

    some_proper_nouns_to_remove = ['dick','ginger','hollywood','jack',\
        'jill','john','karloff','kudrow','orson','peter','tcm','tom',\
        'toni','welles','william','wolheim','nikita']

    # start with the initial list and add to it for movie text work 
    stoplist = nltk.corpus.stopwords.words('english') + more_stop_words +\
        some_proper_nouns_to_remove

# text parsing function for creating text documents 
# there is more we could do for data preparation 
# stemming... looking for contractions... possessives... 
# but we will work with what we have in this parsing function
# if we want to do stemming at a later time, we can use
#     porter = nltk.PorterStemmer()  
# in a construction like this
#     words_stemmed =  [porter.stem(word) for word in initial_words]  
def text_parse(string):
    # replace non-alphanumeric with space 
    temp_string = re.sub('[^a-zA-Z]', '  ', string)    
    # replace codes with space
    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_string = re.sub(stopstring, '  ', temp_string)      
    # replace single-character words with space
    temp_string = re.sub('\s.\s', ' ', temp_string)   
    # convert uppercase to lowercase
    temp_string = temp_string.lower()    
    if REMOVE_STOPWORDS:
        # replace selected character strings/stop-words with space
        for i in range(len(stoplist)):
            stopstring = ' ' + str(stoplist[i]) + ' '
            temp_string = re.sub(stopstring, ' ', temp_string)        
    # replace multiple blank characters with one blank character
    temp_string = re.sub('\s+', ' ', temp_string)    
    return(temp_string)

# -----------------------------------------------
# gather data for 500 negative movie reviews
# -----------------------------------------------
dir_path = '/content/drive/My Drive/422_data_files/movie-reviews-negative'
    
filenames = listdir_no_hidden(path=dir_path)
num_files = len(filenames)

for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_path, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_path)    
print('%d files found' % len(filenames))

# Read data for negative movie reviews
# Data will be stored in a list of lists where the each list represents 
# a document and document is a list of words.
# We then break the text into words.

def read_data(filename):

  with open(filename, encoding='utf-8') as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = text_parse(data)
    data = TreebankWordTokenizer().tokenize(data)  # The Penn Treebank

  return data

negative_documents = []

print('\nProcessing document files under', dir_path)
for i in range(num_files):
    ## print(' ', filenames[i])

    words = read_data(os.path.join(dir_path, filenames[i]))

    negative_documents.append(words)
    # print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    # print('Sample string (Document %d) %s'%(i,words[:50]))

negative_documents[0]

dir_path = '/content/drive/My Drive/422_data_files/movie-reviews-positive'  
filenames = listdir_no_hidden(path=dir_path)
num_files = len(filenames)

for i in range(len(filenames)):
    file_exists = os.path.isfile(os.path.join(dir_path, filenames[i]))
    assert file_exists
print('\nDirectory:',dir_path)    
print('%d files found' % len(filenames))

# Read data for positive movie reviews
# Data will be stored in a list of lists where the each list 
# represents a document and document is a list of words.
# We then break the text into words.

def read_data(filename):

  with open(filename, encoding='utf-8') as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = text_parse(data)
    data = TreebankWordTokenizer().tokenize(data)  # The Penn Treebank

  return data

positive_documents = []

print('\nProcessing document files under', dir_path)
for i in range(num_files):
    ## print(' ', filenames[i])

    words = read_data(os.path.join(dir_path, filenames[i]))

    positive_documents.append(words)
    # print('Data size (Characters) (Document %d) %d' %(i,len(words)))
    # print('Sample string (Document %d) %s'%(i,words[:50]))

# -----------------------------------------------------
# convert positive/negative documents into numpy array
# note that reviews vary from 22 to 1052 words   
# so we use the first 20 and last 20 words of each review 
# as our word sequences for analysis
# -----------------------------------------------------
max_review_length = 0  # initialize
for doc in negative_documents:
    max_review_length = max(max_review_length, len(doc))    
for doc in positive_documents:
    max_review_length = max(max_review_length, len(doc)) 
print('max_review_length:', max_review_length) 

min_review_length = max_review_length  # initialize
for doc in negative_documents:
    min_review_length = min(min_review_length, len(doc))    
for doc in positive_documents:
    min_review_length = min(min_review_length, len(doc)) 
print('min_review_length:', min_review_length) 

# construct list of 1000 lists with 40 words in each list
from itertools import chain
documents = []
for doc in negative_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))    
for doc in positive_documents:
    doc_begin = doc[0:20]
    doc_end = doc[len(doc) - 20: len(doc)]
    documents.append(list(chain(*[doc_begin, doc_end])))

# create list of lists of lists for embeddings
embeddings = []    
for doc in documents:
    embedding = []
    for word in doc:
       embedding.append(limited_index_to_embedding[limited_word_to_index[word]]) 
    embeddings.append(embedding)

# -----------------------------------------------------    
# Check on the embeddings list of list of lists 
# -----------------------------------------------------
# Show the first word in the first document
test_word = documents[0][0]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[0][0][:])

# Show the seventh word in the tenth document
test_word = documents[6][9]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[6][9][:])

# Show the last word in the last document
test_word = documents[999][39]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding[limited_word_to_index[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings[999][39][:])

# -----------------------------------------------------    
# Make embeddings a numpy array for use in an RNN 
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array = np.array(embeddings)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                      np.ones((500), dtype = np.int32)), axis = 0)

# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)

# -----------------------------------------------------    
# Make embeddings a numpy array for use in an RNN 
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array = np.array(embeddings)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                      np.ones((500), dtype = np.int32)), axis = 0)

# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
X_train, X_test, y_train, y_test = \
    train_test_split(embeddings_array, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)

reset_graph()

n_steps = embeddings_array.shape[1]  # number of words per document 
n_inputs = embeddings_array.shape[2]  # dimension of  pre-trained embeddings
n_neurons = 20  # analyst specified number of neurons
n_outputs = 2  # thumbs-down or thumbs-up

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

n_epochs = 50
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        print('\n  ---- Epoch ', epoch, ' ----\n')
        for iteration in range(y_train.shape[0] // batch_size):          
            X_batch = X_train[iteration*batch_size:(iteration + 1)*batch_size,:]
            y_batch = y_train[iteration*batch_size:(iteration + 1)*batch_size]
            print('  Batch ', iteration, ' training observations from ',  
                  iteration*batch_size, ' to ', (iteration + 1)*batch_size-1,)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print('\n  Train accuracy:', acc_train, 'Test accuracy:', acc_test)

"""## Second Lanuage Model with Top 30000 words"""

EVOCABSIZE2 = 30000

def default_factory2():
    return EVOCABSIZE2  # last/unknown-word row in limited_index_to_embedding
# dictionary has the items() function, returns list of (key, value) tuples
limited_word_to_index2 = defaultdict(default_factory2, \
    {k: v for k, v in word_to_index.items() if v < EVOCABSIZE2})

# Select the second EVOCABSIZE rows to the index_to_embedding
limited_index_to_embedding2 = index_to_embedding[0:EVOCABSIZE2,:]
# Set the unknown-word row to be all zeros as previously
limited_index_to_embedding2 = np.append(limited_index_to_embedding2, 
    index_to_embedding[index_to_embedding.shape[0] - 1, :].\
        reshape(1,embedding_dim), 
    axis = 0)

# Verify the new vocabulary: should get same embeddings for test sentence
# Note that a small EVOCABSIZE may yield some zero vectors for embeddings
print('\nTest sentence embeddings from vocabulary of', EVOCABSIZE2, 'words:\n')
for word in words_in_test_sentence:
    word_ = word.lower()
    embedding = limited_index_to_embedding2[limited_word_to_index2[word_]]
    print(word_ + ": ", embedding)

# create list of lists of lists for embeddings
embeddings2 = []    
for doc in documents:
    embedding = []
    for word in doc:
       embedding.append(limited_index_to_embedding2[limited_word_to_index2[word]]) 
    embeddings2.append(embedding)

# -----------------------------------------------------    
# Check on the embeddings list of list of lists 
# -----------------------------------------------------
# Show the first word in the first document
test_word = documents[0][0]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding2[limited_word_to_index2[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings2[0][0][:])

# Show the seventh word in the tenth document
test_word = documents[6][9]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding2[limited_word_to_index2[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings2[6][9][:])

# Show the last word in the last document
test_word = documents[999][39]    
print('First word in first document:', test_word)    
print('Embedding for this word:\n', 
      limited_index_to_embedding2[limited_word_to_index2[test_word]])
print('Corresponding embedding from embeddings list of list of lists\n',
      embeddings2[999][39][:])

# -----------------------------------------------------    
# Make embeddings a numpy array for use in an RNN 
# Create training and test sets with Scikit Learn
# -----------------------------------------------------
embeddings_array2 = np.array(embeddings2)

# Define the labels to be used 500 negative (0) and 500 positive (1)
thumbs_down_up = np.concatenate((np.zeros((500), dtype = np.int32), 
                      np.ones((500), dtype = np.int32)), axis = 0)

# Scikit Learn for random splitting of the data  
from sklearn.model_selection import train_test_split

# Random splitting of the data in to training (80%) and test (20%)  
large_X_train, large_X_test, large_y_train, large_y_test = \
    train_test_split(embeddings_array2, thumbs_down_up, test_size=0.20, 
                     random_state = RANDOM_SEED)

# Delete large numpy array to clear some CPU RAM
del index_to_embedding

"""# Model Experimentation

Done on smaller subset of vocabulary with 10000 words.
"""

from functools import partial

def fit_and_test_model(model, model_name, optimizer, n_epochs):
    """
    This function takes a constructred model architecture then
    compiles, fits to training data, and evaluates the model. 
    
    It outputs a printout of the training accuracies and the evaluated
    test accuracy.
    """
    try:
        name = str(model_name)
    except:
        print("enter a model name")
        pass
    model.compile(optimizer=optimizer,
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=n_epochs, batch_size=20)

    final_train_acc = model.history.history['acc'][-1]

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print('Test accuracy: {}'.format(test_acc))

    # add to dicts

    train_accs[name] = final_train_acc
    test_accs[name] = test_acc

# train acc storage
train_accs = dict()

# test acc storage
test_accs = dict()

# Default settings
adam_optimizer = keras.optimizers.Adam()
n_epochs = 35

from numpy.random import seed
seed(RANDOM_SEED)

from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

# Example
base_model = keras.Sequential()
base_model.add(LSTM(units=20))
base_model.add(Dense(2, activation='softmax'))

# Call function
fit_and_test_model(base_model, 'base_model', adam_optimizer, n_epochs)

"""## SimpleRNN"""

# Default settings
adam_optimizer_simpleRNN_1 = keras.optimizers.Adam(lr=.001)
n_epochs_simpleRNN_1 = 35
activation_simpleRNN_1 = "relu"

# New Settings
adam_optimizer_simpleRNN_2 = keras.optimizers.Adam(lr=.0008)
n_epochs_simpleRNN_2 = 50
activation_simpleRNN_2 = "sigmoid"

from numpy.random import seed
seed(RANDOM_SEED)

from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

# SimpleRNN Default Settings
simpleRNN_1 = keras.Sequential()
simpleRNN_1.add(SimpleRNN(units=20, activation=activation_simpleRNN_1))
simpleRNN_1.add(Dense(2, activation='softmax'))

# Call function
fit_and_test_model(simpleRNN_1, 'simpleRNN_1', adam_optimizer_simpleRNN_1, n_epochs_simpleRNN_1)

# SimpleRNN New Settings
simpleRNN_2 = keras.Sequential()
simpleRNN_2.add(SimpleRNN(units=20, activation=activation_simpleRNN_2))
simpleRNN_2.add(Dense(2, activation='softmax'))

# Call function
fit_and_test_model(simpleRNN_2, 'simpleRNN_2', adam_optimizer_simpleRNN_2, n_epochs_simpleRNN_2)

"""## SimpleRNNCell"""

# Default settings
adam_optimizer_simpleRNNCell_1 = keras.optimizers.Adam(lr=.001)
n_epochs_simpleRNNCell_1 = 35
activation_simpleRNNCell_1 = "relu"

# New Settings
adam_optimizer_simpleRNNCell_2 = keras.optimizers.Adam(lr=.0008)
n_epochs_simpleRNNCell_2 = 50
activation_simpleRNNCell_2 = "sigmoid"

from numpy.random import seed
seed(RANDOM_SEED)

from tensorflow import set_random_seed
set_random_seed(RANDOM_SEED)

# SimpleRNNCell Default Settings
simpleRNNCell_1 = keras.Sequential()
simpleRNNCell_1.add(RNN(
    SimpleRNNCell(units=20, dropout=0.5, activation=activation_simpleRNNCell_1)
))
simpleRNNCell_1.add(Dense(2, activation='softmax'))

# Call function
fit_and_test_model(simpleRNNCell_1, 'simpleRNNCell_1', adam_optimizer_simpleRNNCell_1, n_epochs_simpleRNNCell_1)

# SimpleRNNCell New Settings
simpleRNNCell_2 = keras.Sequential()
simpleRNNCell_2.add(RNN(
    SimpleRNNCell(units=20, dropout=0.5, activation=activation_simpleRNNCell_2)
))
simpleRNNCell_2.add(Dense(2, activation='softmax'))

# Call function
fit_and_test_model(simpleRNNCell_2, 'simpleRNNCell_2', adam_optimizer_simpleRNNCell_2, n_epochs_simpleRNNCell_2)

"""## LSTM"""

base_lstm = keras.Sequential()
base_lstm.add(LSTM(units=20))
base_lstm.add(Dense(2, activation='softmax'))

fit_and_test_model(base_model, 'base_model', adam_optimizer, n_epochs)

# base LSTM model, 20 units, different activation (from tanh)

lstm_model1 = Sequential()
lstm_model1.add(LSTM(units=20, activation='sigmoid'))
lstm_model1.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model1, 'lstm_model1', adam_optimizer, n_epochs)

# base LSTM, 200 units, default activation

lstm_model2 = Sequential()
lstm_model2.add(LSTM(units=200))
lstm_model2.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model2, 'lstm_model2', adam_optimizer, n_epochs)

lstm_model3 = Sequential()
lstm_model3.add(LSTM(units=5))
lstm_model3.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model3, 'lstm_model3', adam_optimizer, n_epochs)

lstm_model4 = Sequential()
lstm_model4.add(LSTM(units = 5, return_sequences=True))
lstm_model4.add(LSTM(units = 5))
lstm_model4.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model4, 'lstm_model4', adam_optimizer, n_epochs)

lstm_model5 = Sequential()
lstm_model5.add(LSTM(units = 200, return_sequences=True))
lstm_model5.add(LSTM(units = 200))
lstm_model5.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model5, 'lstm_model5', adam_optimizer, n_epochs)

# 40 neurons in LSTM
# There was no logic here
lstm_model6 = Sequential()
lstm_model6.add(LSTM(units=40))
lstm_model6.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model6, 'lstm_model6', adam_optimizer, n_epochs)

lstm_model7 = Sequential()
lstm_model7.add(LSTM(units = 5, return_sequences=True))
lstm_model7.add(SimpleRNN(units = 5))
lstm_model7.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model7, 'lstm_model47', adam_optimizer, n_epochs)

# Dropout 0.20 with base model
lstm_model8 = Sequential()
lstm_model8.add(LSTM(units=20, activation='sigmoid'))
lstm_model8.add(Dropout(0.2))
lstm_model8.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model8, 'lstm_model8', adam_optimizer, 50)

# Model 5 with 0.20 dropout

lstm_model9 = Sequential()
lstm_model9.add(LSTM(units = 200, return_sequences=True))
lstm_model9.add(LSTM(units = 200))
lstm_model9.add(Dropout(0.2))
lstm_model9.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model9, 'lstm_model9', adam_optimizer, n_epochs)

# Model 8 with 0.50 dropout
lstm_model10 = Sequential()
lstm_model10.add(LSTM(units=20, activation='sigmoid'))
lstm_model10.add(Dropout(0.5))
lstm_model10.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model10, 'lstm_model10', adam_optimizer, n_epochs)

# Model 5 with 0.50 dropout

lstm_model11 = Sequential()
lstm_model11.add(LSTM(units = 200, return_sequences=True))
lstm_model11.add(LSTM(units = 200))
lstm_model11.add(Dropout(0.5))
lstm_model11.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model11, 'lstm_model11', adam_optimizer, n_epochs)

# Dropout 0.70 with base model
lstm_model12 = Sequential()
lstm_model12.add(LSTM(units=20, activation='sigmoid'))
lstm_model12.add(Dropout(0.7))
lstm_model12.add(Dense(2, activation='softmax'))

fit_and_test_model(lstm_model12, 'lstm_model12', adam_optimizer, 50)

"""## GRU"""

from tensorflow.keras.layers import GRU, GRUCell

base_gru = keras.Sequential()
base_gru.add(GRU(units=20))
base_gru.add(Dense(2, activation='softmax'))

fit_and_test_model(base_gru, 'base_gru', adam_optimizer, n_epochs)

#base layer neurons adjusted
gru2 = keras.Sequential()
gru2.add(GRU(units=50))
gru2.add(Dense(2, activation='softmax'))

fit_and_test_model(gru2, 'gru2', adam_optimizer, n_epochs)

#2 layer neurons adjusted
gru3 = keras.Sequential()
gru3.add(GRU(units=50, return_sequences=True))
gru3.add(GRU(units=50))
gru3.add(Dense(2, activation='softmax'))

fit_and_test_model(gru3, 'gru3', adam_optimizer, n_epochs)

#2 layer base
gru4 = keras.Sequential()
gru4.add(GRU(units=20, return_sequences=True))
gru4.add(GRU(units=20, return_sequences=True))
gru4.add(GRU(units=20))
gru4.add(Dense(2, activation='softmax'))

fit_and_test_model(gru4, 'gru4', adam_optimizer, n_epochs)

"""Not shown but adding more layers or reduce neurons hurt model accuracy, best with 1 layer 20 neurons (attempted 10,20,40,50,100 neurons 1,2,3 layers)"""

# Model architecture
base_gru2 = keras.Sequential()
base_gru2.add(RNN(
    GRUCell(units=20)
))
base_gru2.add(Dense(2, activation='softmax'))

fit_and_test_model(base_gru2, 'base_gru2', adam_optimizer, n_epochs)

# Model architecture
gru2c = keras.Sequential()
gru2c.add(RNN(
    GRUCell(units=50)
))
gru2c.add(Dense(2, activation='softmax'))

fit_and_test_model(gru2c, 'gru2c', adam_optimizer, n_epochs)

# Model architecture
gru3c = keras.Sequential()
gru3c.add(RNN(
    GRUCell(units=50),return_sequences=True))
gru3c.add(RNN(
    GRUCell(units=50),return_sequences=True))
gru3c.add(RNN(
    GRUCell(units=50)))
gru3c.add(Dense(2, activation='softmax'))

fit_and_test_model(gru3c, 'gru3c', adam_optimizer, n_epochs)

# Model architecture
gru4c = keras.Sequential()
gru4c.add(RNN(
    GRUCell(units=20),return_sequences=True))
gru4c.add(RNN(
    GRUCell(units=20),return_sequences=True))
gru4c.add(RNN(
    GRUCell(units=20)))
gru4c.add(Dense(2, activation='softmax'))

fit_and_test_model(gru3c, 'gru3c', adam_optimizer, n_epochs)

"""Not shown but adding more layers or reduce neurons hurt model accuracy, best with 1 layer 50 neurons for GRU cell (attempted 10,20,40,50,100 neurons 1,2,3 layers)

# Experimentation Summary

## Train Accuracies
"""

printer.pprint(train_accs)

"""## Testing Accuracies"""

printer.pprint(test_accs)

"""# 2x2 Experimental Design

* Use LSTM model 10
"""

# Redefine model to take different training and testing inputs
def refit_and_test_model(model, model_name, optimizer, n_epochs, x_train, y_train, x_test, y_test):
    """
    This function takes a constructred model architecture then
    compiles, fits to training data, and evaluates the model. 
    
    It outputs a printout of the training accuracies and the evaluated
    test accuracy.
    """
    try:
        name = str(model_name)
    except:
        print("enter a model name")
        pass
    model.compile(optimizer=optimizer,
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=n_epochs, batch_size=20)

    final_train_acc = model.history.history['acc'][-1]

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy: {}'.format(test_acc))

    # add to dicts

    train_accs[name] = final_train_acc
    test_accs[name] = test_acc

"""## Variable 1: 10000 vs 30000 words used

* 10000 words already defined in experimentation
"""

# model w 10000 words
refit_and_test_model(lstm_model10, 'lstm_10000_20', adam_optimizer, 50, X_train, y_train, X_test, y_test)

# model w 30000 words
refit_and_test_model(lstm_model10, 'lstm_30000_20', adam_optimizer, 50, large_X_train, large_y_train, large_X_test, large_y_test)

"""## Variable 2: Number of Neurons, 20 vs 200"""

# Define the model with 200 neurons
lstm_model_200 = Sequential()
lstm_model_200.add(LSTM(units=200, activation='sigmoid'))
lstm_model_200.add(Dense(2, activation='softmax'))

# 10000 word model, 200 neurons
refit_and_test_model(lstm_model_200, 'lstm_10000_200', adam_optimizer, 50, X_train, y_train, X_test, y_test)

# 30000 word model, 200 neurons
refit_and_test_model(lstm_model_200, 'lstm_30000_200', adam_optimizer, 50, large_X_train, large_y_train, large_X_test, large_y_test)

"""## Results"""

printer.pprint(train_accs)
printer.pprint(test_accs)

"""# Appendix

* https://www.tensorflow.org/api_docs/python
* RNN layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN
* SimpleRNN: https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
* https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell
* LSTM: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
* GRU: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GR
* AbstractRNNCell: https://www.tensorflow.org/api_docs/python/tf/keras/layers/AbstractRNNCell
* Activations: https://www.tensorflow.org/api_docs/python/tf/keras/activations

* 2x2 factorial design example: https://en.wikipedia.org/wiki/Factorial_experiment#Example
* Keras guide to sequential models: https://keras.io/getting-started/sequential-model-guide/
"""