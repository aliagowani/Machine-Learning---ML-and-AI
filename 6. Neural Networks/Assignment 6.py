# -*- coding: utf-8 -*-

# Setup

## Import Modules

# Commented out IPython magic to ensure Python compatibility.
## Import Packages
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow import keras
from sklearn.neural_network import MLPClassifier

# %matplotlib inline

# start total runtime counter
script_start = time.time()

"""## Common Functions

### Get elapsed time in minutes
"""

# return time elapsed in minutes

def elapsed_time(start_time, end_time): # input start and end times just like last time
    min_elapsed = ((end_time - start_time) / 60)
    rounded_min_elapsed = round(min_elapsed, 3)
    return rounded_min_elapsed

"""### Get training time"""

def get_train_acc(fitted_model): # input fitted model to function
    train_acc = fitted_model.history.history['acc'][-1] # returns training acc score of final epoch
    return train_acc

"""### Get average value of a list"""

from statistics import mean
def avg_list(input_list):
    return mean(input_list)

"""### Function examples"""

# Preliminary steps to scale and gather data
mnist = keras.datasets.mnist
(ex_X_train, ex_y_train), (ex_X_test, ex_y_test) = mnist.load_data()

ex_X_train = ex_X_train / 255.0
ex_X_test = ex_X_test / 255.0

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
ex_model = Sequential()
ex_model.add(Flatten(input_shape=(28,28))) # input_layer
ex_model.add(Dense(22, input_dim=28, activation='relu')) 
ex_model.add(Dense(12, activation='relu')) 
ex_model.add(Dense(10, activation='sigmoid')) 

ex_model.compile(optimizer='adam', # 1
              loss='sparse_categorical_crossentropy', # 2
              metrics=['accuracy']) # 3

ex_start_time = time.time() # call time.time to record starting time

# after running this, you can then call the function to grab the training accuracy value
ex_model.fit(ex_X_train, ex_y_train) # variable ex_model is now fitted with training data


ex_end_time = time.time() # same thing for ending time

print(elapsed_time(ex_start_time, ex_end_time))

print(get_train_acc(ex_model)) # Call this function and enter the fitted model

"""## Import dataset"""

mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

class_names = [str(i) for i in range(10)]
class_names

X_train.shape

len(y_train)

"""# Data Prep

## Preprocessing
"""

# Check image first
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scaling
X_train = X_train / 255.0

X_test = X_test / 255.0

# Check images to see if they match up with targets
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

"""# Experimentation

## Model Run Functions
"""

def base_model_run(first_nodes, optimizer):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(first_nodes, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=8)

    return model

"""## Base Model Tests"""

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10) # put a runtime setup here

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('Test accuracy', test_acc)

# baseline model
n_nodes = 128
base_adam = 'adam'

base_model = base_model_run(n_nodes, base_adam)
# train_acc = get_train_acc(base_model)
# print(train_acc)

test_loss, test_acc = base_model.evaluate(X_test, y_test, verbose=2)

"""## Parameter Exploration

### Learning Rates and Optimizers

The following code runs each model 3 times, stores final epoch's training accuracy and testing accuracy. I based it off the example base model detailed above. The code is commented out because iterating through all the models takes about an hour.
"""

def loop_model_runs(model_name, optimizer, acc_list):
    print("Start {} run:".format(model_name))
    for i in range(len(learning_rates)):
        print("Start learning rate: {}".format(learning_rates[i]))
        model = base_model_run(128, optimizer)
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        print("Test accuracy: {}".format(test_acc))
        acc_list.append(test_acc)
        print("Finished model for learning rate {}\n".format(learning_rates[i]))
    print("Finished {} run.".format(model_name))

"""#### Adam"""

# adam_train_acc_dict = dict()
# adam_test_acc_dict = dict()
# print("Start Adam optimizer:")
# for n in range(1,4):
#     for i, learning_rate in enumerate(learning_rates):
#         print("Start learning rate: {}".format(learning_rates[i]))
#         optimizer = keras.optimizers.Adam(lr=learning_rates[i])
#         model = base_model_run(128, optimizer)
#         try:
#             adam_train_acc_dict[n][learning_rate] = get_train_acc(model)
#         except:
#             adam_train_acc_dict[n] = {learning_rate: get_train_acc(model)}
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print("Test accuracy: {}".format(test_acc))
#         adam_test_acc_dict[n] = {learning_rate: test_acc}
#         print("Finished model for learning rate {}\n".format(learning_rates[i]))

# print("\nFinished Adam optimizer.")

"""#### SGD"""

# sgd_train_acc_dict = dict()
# sgd_test_acc_dict = dict()
# print("Start SGD optimizer:")
# for n in range(1,4):
#     for i, learning_rate in enumerate(learning_rates):
#         print("Start learning rate: {}".format(learning_rates[i]))
#         optimizer = keras.optimizers.SGD(lr=learning_rates[i])
#         model = base_model_run(128, optimizer)
#         try:
#             sgd_train_acc_dict[n][learning_rate] = get_train_acc(model)
#         except:
#             sgd_train_acc_dict[n] = {learning_rate: get_train_acc(model)}
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print("Test accuracy: {}".format(test_acc))
#         sgd_test_acc_dict[n] = {learning_rate: test_acc}
#         print("Finished model for learning rate {}\n".format(learning_rates[i]))

# print("\nFinished SGD optimizer.")

"""#### Nadam"""

# nadam_train_acc_dict = dict()
# nadam_test_acc_dict = dict()
# print("Start Nadam optimizer:")
# for n in range(1,4):
#     for i, learning_rate in enumerate(learning_rates):
#         print("Start learning rate: {}".format(learning_rates[i]))
#         optimizer = keras.optimizers.Nadam(lr=learning_rates[i])
#         model = base_model_run(128, optimizer)
#         try:
#             nadam_train_acc_dict[n][learning_rate] = get_train_acc(model)
#         except:
#             nadam_train_acc_dict[n] = {learning_rate: get_train_acc(model)}
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print("Test accuracy: {}".format(test_acc))
#         nadam_test_acc_dict[n] = {learning_rate: test_acc}
#         print("Finished model for learning rate {}\n".format(learning_rates[i]))

# print("\nFinished Nadam optimizer.")

"""#### Adagrad"""

# adagrad_train_acc_dict = dict()
# adagrad_test_acc_dict = dict()
# print("Start Adagrad optimizer:")
# for n in range(1,4):
#     for i, learning_rate in enumerate(learning_rates):
#         print("Start learning rate: {}".format(learning_rates[i]))
#         optimizer = keras.optimizers.Adagrad(lr=learning_rates[i])
#         model = base_model_run(128, optimizer)
#         try:
#             adagrad_train_acc_dict[n][learning_rate] = get_train_acc(model)
#         except:
#             adagrad_train_acc_dict[n] = {learning_rate: get_train_acc(model)}
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print("Test accuracy: {}".format(test_acc))
#         adagrad_test_acc_dict[n] = {learning_rate: test_acc}
#         print("Finished model for learning rate {}\n".format(learning_rates[i]))

# print("\nFinished Adagrad optimizer.")

"""#### Adamax"""

# adamax_train_acc_dict = dict()
# adamax_test_acc_dict = dict()
# print("Start Adamax optimizer:")
# for n in range(1,4):
#     for i, learning_rate in enumerate(learning_rates):
#         print("Start learning rate: {}".format(learning_rates[i]))
#         optimizer = keras.optimizers.Adamax(lr=learning_rates[i])
#         model = base_model_run(128, optimizer)
#         try:
#             adamax_train_acc_dict[n][learning_rate] = get_train_acc(model)
#         except:
#             adamax_train_acc_dict[n] = {learning_rate: get_train_acc(model)}
#         test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
#         print("Test accuracy: {}".format(test_acc))
#         adamax_test_acc_dict[n] = {learning_rate: test_acc}
#         print("Finished model for learning rate {}\n".format(learning_rates[i]))

# print("\nFinished Adamax optimizer.")

"""#### Conclusion

The learning rate and optimizer can be adjusted to achieve the same results. We can pick and choose the learning rate simply by what optimizer we end up choosing.

### Layers and Numbers of Nodes
"""

n_outputs = 10

#adam model - 1 layer, 128 nodes
model1_train_acc = list()
model1_test_acc = list()
model1_elapsed_times = list()

for i in range(3):
    adam_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    adam_model.compile(optimizer='adam', # 1
                loss='sparse_categorical_crossentropy', # 2
                metrics=['accuracy']) # 3

    adam_start_time = time.time() # call time.time to record starting time
    adam_model.fit(X_train, y_train, epochs=10) # put a runtime setup here

    adam_end_time = time.time() # same thing for ending time
    test_loss, test_acc = adam_model.evaluate(X_test, y_test, verbose=2)
    duration = elapsed_time(adam_start_time, adam_end_time)
    train_acc= get_train_acc(adam_model)

    model1_train_acc.append(train_acc)
    model1_test_acc.append(test_acc)
    model1_elapsed_times.append(duration)

# print ('Adam Base Model')
# #print('elapsed time:', elasped_time(adam_start_time, adam_end_time))
# #print('Training accuracy:', get_train_acc(adam_model)) # Call this function and enter the fitted model
# print('Elapsed time:', duration)
# print('Training accuracy:', train_acc) # Call this function and enter the fitted model
# print('Test accuracy', test_acc)
# metrics['adam_model'] = [1, 128, duration, train_acc, test_acc]

print(avg_list(model1_train_acc))
print(avg_list(model1_test_acc))
print(avg_list(model1_elapsed_times))

"""This model looks underfit."""

#barry model - 1 layer, 356 nodes
model2_train_acc = list()
model2_test_acc = list()
model2_elapsed_times = list()
for i in range(3):
    barry_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(356, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    barry_model.compile(optimizer='adam', # 1
                loss='sparse_categorical_crossentropy', # 2
                metrics=['accuracy']) # 3

    barry_start_time = time.time() # call time.time to record starting time
    barry_model.fit(X_train, y_train, epochs=10) # put a runtime setup here

    barry_end_time = time.time() # same thing for ending time
    test_loss, test_acc = barry_model.evaluate(X_test, y_test, verbose=2)
    duration = elapsed_time(barry_start_time, barry_end_time)
    train_acc= get_train_acc(barry_model)

    model2_train_acc.append(train_acc)
    model2_test_acc.append(test_acc)
    model2_elapsed_times.append(duration)
# print ('Barry Base Model')
# print('Elapsed time:', duration)
# print('Training accuracy:', train_acc) # Call this function and enter the fitted model
# print('Test accuracy', test_acc)
# metrics['barry_model'] = [1, 356, duration, train_acc, test_acc]

print(avg_list(model2_train_acc))
print(avg_list(model2_test_acc))
print(avg_list(model2_elapsed_times))

"""This model seems very overfit."""

#charley model - 1 layer, 500 nodes
model3_train_acc = list()
model3_test_acc = list()
model3_elapsed_times = list()
for i in range(3):
    charley_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(500, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    charley_model.compile(optimizer='adam', # 1
                loss='sparse_categorical_crossentropy', # 2
                metrics=['accuracy']) # 3

    charley_start_time = time.time() # call time.time to record starting time
    charley_model.fit(X_train, y_train, epochs=10) # put a runtime setup here

    charley_end_time = time.time() # same thing for ending time
    test_loss, test_acc = charley_model.evaluate(X_test, y_test, verbose=2)
    duration = elapsed_time(charley_start_time, charley_end_time)
    train_acc= get_train_acc(charley_model)

    model3_train_acc.append(train_acc)
    model3_test_acc.append(test_acc)
    model3_elapsed_times.append(duration)
# print ('Charley Base Model')
# print('Elapsed time:', duration)
# print('Training accuracy:', train_acc) # Call this function and enter the fitted model
# print('Test accuracy', test_acc)
# metrics['charley_model'] = [1, 500, duration, train_acc, test_acc]

print(avg_list(model3_train_acc))
print(avg_list(model3_test_acc))
print(avg_list(model3_elapsed_times))

"""This is a medium, but it's still overfit.

### Optimizers Refinement

#### Adam
"""

adam_train_accs = list()
adam_test_accs = list()
adam_elapsed_times = list()

for i in range(3):
    adam_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    adam_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']) 
    adam_start_time = time.time() 
    adam_model.fit(X_train, y_train, epochs=10) 
    adam_end_time = time.time()

    adam_test_loss, adam_test_acc = adam_model.evaluate(X_test, y_test, verbose=2)

    adam_train_accs.append(get_train_acc(adam_model))
    adam_test_accs.append(adam_test_acc)
    adam_elapsed_times.append(elapsed_time(adam_start_time, adam_end_time))

print(mean(adam_train_accs))
print(mean(adam_test_accs))
print(mean(adam_elapsed_times))

"""#### SGD"""

sgd_train_accs = list()
sgd_test_accs = list()
sgd_elapsed_times = list()

for i in range(3):
    sgd_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    sgd_model.compile(optimizer='SGD',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    sgd_start_time = time.time() 
    sgd_model.fit(X_train, y_train, epochs=10) 
    sgd_end_time = time.time() 

    sgd_test_loss, sgd_test_acc = sgd_model.evaluate(X_test, y_test, verbose=2)

    sgd_train_accs.append(get_train_acc(sgd_model))
    sgd_test_accs.append(sgd_test_acc)
    sgd_elapsed_times.append(elapsed_time(sgd_start_time, sgd_end_time))

print(mean(sgd_train_accs))
print(mean(sgd_test_accs))
print(mean(sgd_elapsed_times))

"""#### RMSProp"""

rms_train_accs = list()
rms_test_accs = list()
rms_elapsed_times = list()

for i in range(3):
    rms_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    rms_model.compile(optimizer='RMSProp',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    rms_start_time = time.time() 
    rms_model.fit(X_train, y_train, epochs=10) 
    rms_end_time = time.time() 

    rms_test_loss, rms_test_acc = rms_model.evaluate(X_test, y_test, verbose=2)

    rms_train_accs.append(get_train_acc(rms_model))
    rms_test_accs.append(rms_test_acc)
    rms_elapsed_times.append(elapsed_time(rms_start_time, rms_end_time))

print(mean(rms_train_accs))
print(mean(rms_test_accs))
print(mean(rms_elapsed_times))

"""#### Adagrad"""

ada_train_accs = list()
ada_test_accs = list()
ada_elapsed_times = list()

for i in range(3):
    ada_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])

    ada_model.compile(optimizer='Adagrad', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    
    ada_start_time = time.time() 
    ada_model.fit(X_train, y_train, epochs=10) 
    ada_end_time = time.time() 

    ada_test_loss, ada_test_acc = ada_model.evaluate(X_test, y_test, verbose=2)

    ada_train_accs.append(get_train_acc(ada_model))
    ada_test_accs.append(ada_test_acc)
    ada_elapsed_times.append(elapsed_time(ada_start_time, ada_end_time))

print(mean(ada_train_accs))
print(mean(ada_test_accs))
print(mean(ada_elapsed_times))

"""#### Nadam"""

nadam_train_accs = list()
nadam_test_accs = list()
nadam_elapsed_times = list()

for i in range(3):
    nadam_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)), # input layer
        keras.layers.Dense(128, activation='relu'), # 1st hidden layer
        keras.layers.Dense(n_outputs, activation='softmax') # output layer
    ])
    nadam_model.compile(optimizer='Nadam',
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']) 

    nadam_start_time = time.time() 
    nadam_model.fit(X_train, y_train, epochs=10) 
    nadam_end_time = time.time() 

    nadam_test_loss, nadam_test_acc = nadam_model.evaluate(X_test, y_test, verbose=2)

    nadam_train_accs.append(get_train_acc(nadam_model))
    nadam_test_accs.append(nadam_test_acc)
    nadam_elapsed_times.append(elapsed_time(nadam_start_time, nadam_end_time))

print(mean(nadam_train_accs))
print(mean(nadam_test_accs))
print(mean(nadam_elapsed_times))

"""### Number of Epochs, Drop Layers

#### Model: Flatten, Dense, Drop, 2-layers, Epoch: 20
"""

#Prepare the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
X_train /= 255
X_test /= 255
print('x_train shape:', X_train.shape)
print('Number of images in x_train', X_train.shape[0])
print('Number of images in x_test', X_test.shape[0])

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

model_drop1_train_accs = list()
model_drop1_test_accs = list()
model_drop1_elapsed_times = list()

for i in range(3):
    model_flat_dense_drop = Sequential()
    model_flat_dense_drop.add(MaxPooling2D(pool_size=(2, 2)))
    model_flat_dense_drop.add(Flatten())
    model_flat_dense_drop.add(Dense(784, activation=tf.nn.relu))
    model_flat_dense_drop.add(Dropout(0.2))
    model_flat_dense_drop.add(Dense(10,activation=tf.nn.softmax))

    model_flat_dense_drop.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])

    model_flat_dense_drop_start_time = time.time()

    model_flat_dense_drop.fit(x=X_train,y=y_train, epochs=20)

    model_flat_dense_drop_end_time = time.time()

    model_flat_dense_drop_test_loss, model_flat_dense_drop_test_acc = model_flat_dense_drop.evaluate(X_test, y_test, verbose=2)

    model_drop1_train_accs.append(get_train_acc(model_flat_dense_drop))
    model_drop1_test_accs.append(model_flat_dense_drop_test_acc)
    model_drop1_elapsed_times.append(elasped_time(model_flat_dense_drop_start_time, model_flat_dense_drop_end_time))

print(mean(model_drop1_train_accs))
print(mean(model_drop1_test_accs))
print(mean(model_drop1_elapsed_times))

"""This many epochs looks like it overfits data.

#### Model_2: Flatten, Dense, Drop, 2-layers, Epoch: 6
"""

# #Prepare the data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
# X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# input_shape = (28, 28, 1)
# # Making sure that the values are float so that we can get decimal points after division
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
# X_train /= 255
# X_test /= 255
# print('x_train shape:', X_train.shape)
# print('Number of images in x_train', X_train.shape[0])
# print('Number of images in x_test', X_test.shape[0])

# Start with empty lists to store each type of data
model_drop2_train_accs = list()
model_drop2_test_accs = list()
model_drop2_elapsed_times = list()

for i in range(3):
    model_flat_dense_drop2 = Sequential()
    model_flat_dense_drop2.add(MaxPooling2D(pool_size=(2, 2)))
    model_flat_dense_drop2.add(Flatten())
    model_flat_dense_drop2.add(Dense(256, activation=tf.nn.relu))
    model_flat_dense_drop2.add(Dropout(0.1))
    model_flat_dense_drop2.add(Dense(10,activation=tf.nn.softmax))
    model_flat_dense_drop2.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])
    model_flat_dense_drop2_start_time = time.time()

    model_flat_dense_drop2.fit(x=X_train,y=y_train, epochs=6)

    model_flat_dense_drop2_end_time = time.time()
    model_flat_dense_drop2_test_loss, model_flat_dense_drop2_test_acc = model_flat_dense_drop2.evaluate(X_test, y_test, verbose=2)

    model_drop2_train_accs.append(get_train_acc(model_flat_dense_drop2))
    model_drop2_test_accs.append(model_flat_dense_drop2_test_acc)
    model_drop2_elapsed_times.append(elasped_time(model_flat_dense_drop2_start_time, model_flat_dense_drop2_end_time))

print(mean(model_drop2_train_accs))
print(mean(model_drop2_test_accs))
print(mean(model_drop2_elapsed_times))

"""### Summary of results

Details to include:
* Number of epochs
* Optimizer
* Hidden layers
* Number of nodes in hidden layer
* Dropout rate
* Training accuracy score
* Testing accuracy score

Model 1: 
* 10 epochs
* Adam optimizer
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.92731667
* test: 0.95856667
* elapsed time: 0.10733333333333334

Model 2: 
* 10 epochs
* Adam
* 1 hidden layer
* 356 nodes
* no dropout
* train: 0.99658334
* test: 0.9814
* elapsed time: 1.3943333333333332

Model 3: 
* 10 epochs
* Adam
* 1 hidden layer
* 500 nodes
* no dropout
* train: 0.9961778
* test: 0.98163337
* elapsed time: 1.733

Model 4: 
* 10 epochs
* Adam
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.9951111
* test: 0.9766
* elapsed time: 1.07

Model 5:
* 10 epochs
* SGD
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.9532556
* test: 0.9528
* elapsed time: 0.977

Model 6:
* 10 epochs
* RMSProp
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.9909889
* test: 0.9783
* elapsed time: 1.3083333333333333

Model 7:
* 10 epochs
* Adagrad
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.92221665
* test: 0.92433333
* elapsed time: 1.059

Model 8:
* 10 epochs
* Nadam
* 1 hidden layer
* 128 nodes
* no dropout
* train: 0.99539447
* test: 0.9781
* elapsed time: 1.795

Model 9: 
* 20 epochs
* Adam
* 1 hidden layer
* 784 nodes
* 0.2 dropout
* train: 0.99375
* test: 0.9777333333333333
* elapsed time: 4.678

Model 9: 
* 6 epochs
* Adam
* 1 hidden layer 
* 256 nodes
* 0.1 dropout
* train: 0.9780611111111112
* test: 0.9748
* elapsed time: 0.7896666666666667
"""

summary_dicts = [
    {
        'epochs': 10,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(model1_train_acc),
        'mean_test_acc': mean(model1_test_acc),
        'mean_elapsed_time': mean(model1_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 356,
        'dropout_rate': 0,
        'mean_train_acc': mean(model2_train_acc),
        'mean_test_acc': mean(model2_test_acc),
        'mean_elapsed_time': mean(model2_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 500,
        'dropout_rate': 0,
        'mean_train_acc': mean(model3_train_acc),
        'mean_test_acc': mean(model3_test_acc),
        'mean_elapsed_time': mean(model3_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(adam_train_accs),
        'mean_test_acc': mean(adam_test_accs),
        'mean_elapsed_time': mean(adam_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'SGD',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(sgd_train_accs),
        'mean_test_acc': mean(sgd_test_accs),
        'mean_elapsed_time': mean(sgd_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'RMSProp',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(rms_train_accs),
        'mean_test_acc': mean(rms_test_accs),
        'mean_elapsed_time': mean(rms_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'Adagrad',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(ada_train_accs),
        'mean_test_acc': mean(ada_test_accs),
        'mean_elapsed_time': mean(ada_elapsed_times)
    },
    {
        'epochs': 10,
        'optimizer': 'Nadam',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0,
        'mean_train_acc': mean(nadam_train_accs),
        'mean_test_acc': mean(nadam_test_accs),
        'mean_elapsed_time': mean(nadam_elapsed_times)
    },
    {
        'epochs': 20,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 784,
        'dropout_rate': 0.2,
        'mean_train_acc': mean(model_drop1_train_accs),
        'mean_test_acc': mean(model_drop1_test_accs),
        'mean_elapsed_time': mean(model_drop1_elapsed_times)
    },
    {
        'epochs': 6,
        'optimizer': 'Adam',
        'hidden_layers': 1,
        'num_nodes': 128,
        'dropout_rate': 0.1,
        'mean_train_acc': mean(model_drop2_train_accs),
        'mean_test_acc': mean(model_drop2_test_accs),
        'mean_elapsed_time': mean(model_drop2_elapsed_times)
    }
]

df = pd.DataFrame.from_dict(summary_dict)
df

"""# Final Model"""

# Reload and rescale data
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0

X_test = X_test / 255.0

final_optimizer = keras.optimizers.Adamax(lr=0.001)

final_train_accs = list()
final_test_accs = list()
final_elapsed_times = list()

for i in range(3):
    final_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')                                
    ])
    final_model.compile(optimizer=final_optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
    start = time.time()
    final_model.fit(X_train, y_train, epochs=8)
    end = time.time()
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    final_train_accs.append(get_train_acc(final_model))
    final_test_accs.append(test_acc)
    final_elapsed_times.append(elapsed_time(start, end))

summary_dicts.append({
    'epochs': 8,
    'optimizer': 'Adamax',
    'hidden_layers': 1,
    'num_nodes': 128,
    'dropout_rate': 0,
    'mean_train_acc': mean(final_train_accs),
    'mean_test_acc': mean(final_test_accs),
    'mean_elapsed_time': mean(final_elapsed_times)
})

"""# Summary"""

final_df = pd.DataFrame.from_dict(summary_dicts)
final_df

"""# Code Demos from Tutorials

## Keras

### Load and inspect data
"""

mnist = keras.datasets.mnist # load dataset from keras library

(X_train, y_train), (X_test, y_test) = mnist.load_data()

class_names = [str(i) for i in range(10)] # A list of the targets in string form
class_names

X_train.shape

len(y_train)

X_test.shape

# Inspect image
plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.grid(False)
plt.show()

"""### Preprocess data"""

# Scale to range 0 to 1

X_train = X_train / 255.0

X_test = X_test / 255.0

"""Process the training and testing data the same way"""

# Inspect first 25 images in training data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

"""### Building the Model

#### Model Architecture

This is something we can easily modify in Keras.
"""

n_outputs = 10

adam_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # input layer
    keras.layers.Dense(128, activation='relu'), # 1st hidden layer
    keras.layers.Dense(n_outputs, activation='softmax') # output layer
])

"""1. `tf.keras.layers.Flatten` transforms a 28x28 px and lines up each individual pixel into a line. 

2. `keras.layers.Dense(128, activation='relu')` 128 for 128 nodes. `relu` is the activator.

3. Final layer is the output layer. It returns the probabilities of model predictions. For instance, if you fed it an image to predict, it returns 10 probabilities corresponding to the labels (Numbers 0-9), the largest one being the prediction.
"""

model.compile(optimizer='adam', # 1
              loss='sparse_categorical_crossentropy', # 2
              metrics=['accuracy']) # 3

model.fit(X_train, y_train, epochs=10) # put a runtime setup here

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('Test accuracy', test_acc)

"""1. Adam optimizer

2. Loss function

3. Metric of evaluation

#### Model Fitting
"""

model.fit(X_train, y_train, epochs=10) # put a runtime setup here

"""#### Model Testing"""

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('Test accuracy', test_acc)

"""The test set scores lower than the training set. How can we relax the model, so it doesn't overfit now?

# Assessing Prediction Results

### Assessing Prediction Results
"""

predictions = model.predict(X_test)

predictions[0]

"""This returns an array of 10 probabilities which are according to the labels."""

np.argmax(predictions[0])

"""This returns the index of the number with the highest probability."""

# Check if the prediction was correct.
y_test[0]

"""#### Follow ups: We can graph the probability distributions of some murkier predictions to have a look as to which numbers are being mixed together.

### Summary of points we can modify

* Model architecture 
    * number of hidden layers - Ali
    * number of nodes in each hidden layer - Vani
    * activation algorithm
* Optimizer - Noah
* Number of epochs
* Learning rate (built into optimizer) - Ben
    * https://keras.io/optimizers/

## TensorFlow

### Load and Process Data
"""

# Load data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# The number of data points in each set
n_train = mnist.train.num_examples # 55000 
n_validation = mnist.validation.num_examples # 5000
n_test = mnist.test.num_examples # 10000

"""### Building Model Architecture"""

# Define NN Architecture
n_input = 784 # input layer, (28x28 pixels)
n_hidden1 = 512 # 1st hidden layer
n_hidden2 = 256 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 10 # number of outputs, (digits 0-9)

# Parameters
learning_rate = 2e-5
n_interations = 1000
batch_size = 128
dropout = 0.3 # Helps prevent overfitting

# Build TF Graph
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) # controls dropout rate

# Weights and biases
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1' : tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2' : tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3' : tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out' : tf.Variable(tf.constant(0.1, shape=[n_output]))
}

"""Documentation: 

* https://www.tensorflow.org/guide/variable
* https://docs.w3cub.com/tensorflow~python/tf/truncated_normal/

Guided text from article: 

"Since the values are optimized during training, we could set them to zero for now. But the initial value actually has a significant impact on the final accuracy of the model. We’ll use random values from a truncated normal distribution for the weights. We want them to be close to zero, so they can adjust in either a positive or negative 
ection, and slightly different, so they generate different errors. This will ensure that the model learns something useful."

"For the bias, we use a small constant value to ensure that the tensors activate in the intial stages and therefore contribute to the propagation. The weights and bias tensors are stored in dictionary objects for ease of access."
"""

# Inputting operations 
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

# optimization algorithm - gradient descent
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=Y, logits=output_layer
    )
)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Define evaluation method
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1)) # comparing direct accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

"""### Model Fitting"""

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Training in mini batches
for i in range(n_interations):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={
        X: batch_x, Y: batch_y, keep_prob: dropout
    })

    # print loss and accuracy per mini batch
    if i % 100 == 0:
        minibatch_loss, minibatch_accuracy = sess.run(
            [cross_entropy, accuracy],
            feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0}
        )
        print(
            "Iteration", str(i),
            '\t| Loss = ', str(minibatch_loss),
            '\t| Accuracy =', str(minibatch_accuracy)
        )

"""### Model Testing"""

# testing
test_accuracy = sess.run(accuracy, feed_dict={
    X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0
})
print('\nAccuracy on test set: ', test_accuracy)

"""Ending note: In this tutorial you successfully trained a neural network to classify the MNIST dataset with around 92% accuracy and tested it on an image of your own. Current state-of-the-art research achieves around 99% on this same problem, using more complex network architectures involving convolutional layers.

### SK-Learn MLP Classifier

This one takes much, much longer to run than the others.

Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
"""

# Reload dataset first after variables were reset and rearranged
mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

mlp = MLPClassifier(hidden_layer_sizes=(100, 4),
                    activation='relu', # default value
                    solver='adam', # default value
                    batch_size='auto', # default value
                    learning_rate='adaptive',
                    random_state=1)

mlp.fit(X_train, y_train)

mlp_preds = mlp.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, mlp_preds)

"""# End runtime"""

script_end = time.time()

total_runtime = elapsed_time(script_start, script_end)

print("Total script runtime: {}".format(total_runtime))

"""# Appendix

* All variables

# Notes

* Any caveats or gaps in logic we should all be aware of
    * This may not be possible to account for at all since NN are such black boxes.

## Things I want to do (Ben)

* Play with number of layers
* Play with learning rate
* Figure out a systematic approach to gradient descent, a grid search function
* Record runtimes
* Play with initialization methods
* tensorboard if we have time
* scikit-learn mlp classifier

## Questions - I've already emailed Jordan about these.

* How do we systemically tune a model like this using something like grid search?
    * There are so many parameters, and tuning each one makes a different mess especially with deeper networks.
* In another dataset, how do we adjust for the shape of data?
* In TF tutorial, what are the weights and biases doing? 
    * how are they included into the model to begin with?
    * is there a simpler tutorial I can adjust myself? 
* TF documentation is loaded with keras now instead of lower level TF code. Does this mean we should just continue building and iterating with Keras code?

## Things to remember to add

* Timing functions for model fitting
"""
