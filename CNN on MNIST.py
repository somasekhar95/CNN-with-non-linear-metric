from __future__ import print_function
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report,confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
from keras import backend as K
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from numpy import *



mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = mnist.train.images
print('X_train shape:',x_train.shape)
y_train = mnist.train.labels
print('y_train shape:',y_train.shape)
x_test = mnist.test.images
print('X_test shape:',x_test.shape)
y_test = mnist.test.labels
print('y_test shape:',y_test.shape)



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#print(y_train.shape)

#y_test = keras.utils.to_categorical(y_test, num_classes)
#print(y_test.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])

#model.add(Conv2D(32, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(Conv2D(64, (5, 5), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.25))
#model.add(Dense(num_classes, activation='softmax'))
#model.compile(loss=keras.losses.logcosh,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


Y_pred = model.predict(x_test)
print('Y_pred shape:',Y_pred.shape)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print('y_pred shape', y_pred.shape)
print(y_pred)

print('y test shape:',y_test.shape)
print(y_test)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(0)', 'class 1(1)', 'class 2(2)','class 3(3)','class 4(4)', 'class 5(5)', 'class 6(6)','class 7(7)'
                ,'class 8(8)', 'class 9(9)']
					
#print(classification_report(np.argmax(y_test,axis=0), y_pred,target_names=target_names))

#print(confusion_matrix(Y_test, y_pred))




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],'.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix

#y_pred = keras.utils.to_categorical(y_pred, num_classes)
indices = []

for i in range(0,10000):
    array = []
    array = y_test[i]
    #indices.append(array.index(1))
    occ = 0
    for j in array:
        occ +=1
        if j == 1:
            break
    indices.append(occ-1)

cnf_matrix = (confusion_matrix(indices, y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.figure()
# Plot normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True,
                      title='Normalized confusion matrix')
plt.figure()
plt.show()
