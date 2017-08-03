from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from sklearn.metrics import classification_report,confusion_matrix
import itertools
from math import *

import numpy as np
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

path1   ="C:/Users/sekha/Desktop/DR/python CNN/CMU/face_images2"
path2   ="C:/Users/sekha/Desktop/DR/python CNN/CMU/resized_face_images2"

for i in path2:
    print(i)
listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)
img_rows = 28
img_cols = 28

for file in listing:
    im = Image.open(path1+ '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(path2+ '\\' + file, "JPEG")

imlist = os.listdir(path2)
im1 = array(Image.open('C:/Users/sekha/Desktop/DR/python CNN/CMU/resized_face_images2'+'\\'+imlist[0]))
m,n = im1.shape[0:2]
imnbr=len(imlist)

immatrix = array([array(Image.open('C:/Users/sekha/Desktop/DR/python CNN/CMU/resized_face_images2'+'\\'+im2)).flatten()
                  for im2 in imlist],'f')
label = np.ones((num_samples,),dtype=int)
len(label)
count = 0
for i in range(0,20):
    label[(i*32):((i+1)*32)] = count
    count+=1

data,label = shuffle(immatrix,label,random_state=2)
train_data = [data,label]
img= immatrix[167].reshape(img_rows,img_cols)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)



batch_size=32
no_classes=20

img_channels=1
no_filters = 32
no_pool = 2
no_conv = 3

(X,y) = (train_data[0],train_data[1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state =4)

print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print(y_train)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)


X_train = X_train.reshape(X_train.shape[0],img_rows,img_cols,1)
X_test = X_test.reshape(X_test.shape[0],img_rows,img_cols,1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print ('X_train shape:', X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')

Y_train = np_utils.to_categorical(y_train,no_classes)
print(Y_train)
Y_test = np_utils.to_categorical(y_test,no_classes)

i=100
plt.imshow(X_train[i,0],interpolation='nearest')
print("label:",Y_train[i,:])

num_classes = 20
num_epoch = 20

model = Sequential()
model.add(Convolution2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Convolution2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=16, nb_epoch=num_epoch)
#model.fit(X_train, y_train, batch_size=32, nb_epoch=10,verbose=1, validation_split=0.3)

score = model.evaluate(X_test, Y_test, verbose=0)

print('Test Loss:', score[0])
print('Test accuracy:', score[1])
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])
  

Y_pred = model.predict(X_test)
print('Y_pred shape:',Y_pred.shape)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print('y_pred shape', y_pred.shape)
print(y_pred)

print('y test shape:',y_test.shape)
print(y_test)
#y_pred = model.predict_classes(X_test)
#print(y_pred)
target_names = ['class 0(ani22)', 'class 1(at33)', 'class 2(boland)','class 3(bpm)','class 4(ch4f)', 'class 5(cheyer)', 'class 6(choon)','class 7(danieln)'
                ,'class 8(glickman)', 'class 9(karyadi)', 'class 10(kawamura)','class 11(kk49)','class 12(megak)', 'class 13(mitchell)', 'class 14(night)','class 15(phoebe)'
                ,'class 16(saavik)', 'class 17(steffi)', 'class 18(sz24)','class 19(tammo)']
					
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
        plt.text(j, i, format(cm[i, j],'.1f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(y_test, y_pred))

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
    




