from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from sklearn.metrics import classification_report,confusion_matrix
import itertools

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


import numpy as np
import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from sklearn.model_selection import validation_curve
from sklearn.linear_model import Ridge

#path1   ="C:/Users/sekha/Desktop/DR/python CNN/CMU/face_images2"
#path2   ="C:/Users/sekha/Desktop/DR/python CNN/CMU/resized_face_images2"

path1   ="C:/Users/sekha/Desktop/DR/python CNN/oxford flowers/flowers"
path2   ="C:/Users/sekha/Desktop/DR/python CNN/oxford flowers/resized_fl"

listing = os.listdir(path1)
num_samples=size(listing)
print(num_samples)
img_rows = 100
img_cols = 100

for file in listing:
    im = Image.open(path1+ '\\' + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(path2+ '\\' + file, "JPEG")

imlist = os.listdir(path2)
im1 = array(Image.open('C:/Users/sekha/Desktop/DR/python CNN/oxford flowers/resized_fl'+'\\'+imlist[0]))
m,n = im1.shape[0:2]
imnbr=len(imlist)

immatrix = array([array(Image.open('C:/Users/sekha/Desktop/DR/python CNN/oxford flowers/resized_fl'+'\\'+im2)).flatten()
                  for im2 in imlist],'f')
label = np.ones((num_samples,),dtype=int)
count = 0
for i in range(0,17):
    label[(i*80):((i+1)*80-1)] = count
    count+=1

data,label = shuffle(immatrix,label,random_state=2)
train_data = [data,label]
img= immatrix[167].reshape(img_rows,img_cols)
plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)



#batch_size=32
no_classes=17 

img_channels=1
no_filters = 32
no_pool = 2
no_conv = 3

(X,y) = (train_data[0],train_data[1])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state =4)

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
Y_test = np_utils.to_categorical(y_test,no_classes)

#i=100
#plt.imshow(X_train[i,0],interpolation='nearest')
#print("label:",Y_train[i,:])

num_classes = 17
num_epoch = 5


model = Sequential()
model.add(Convolution2D(32, 3, 3,input_shape=(100,100,1), activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(Convolution2D(128, 3, 3, activation='relu'))
#model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(17, activation='softmax'))

#optimizer = Nadam(lr=0.002,
#                  beta_1=0.9,
#                  beta_2=0.999,
#                  epsilon=1e-08,
#                  schedule_decay=0.004)

model.compile(loss='kullback_leibler_divergence',
              optimizer='SGD',
              metrics=['accuracy'])


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
print('datagen fit')
datagen.fit(X_train)


for e in range(0,10):
    print('Epoch', e)
    batches = 0
    for X_b, Y_b in datagen.flow(X_train, Y_train, batch_size=80):
        model.fit(X_b, Y_b)
        batches += 1
        print(batches)
        if batches >= len(X_train) / 80:
            break


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

print('X train shape:',X_train.shape)
print(X_train)
print('Y train shape:',Y_train.shape)
print(Y_train)
print('X test shape:',X_test.shape)
print(X_test)
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])
print(Y_test)
print(Y_test.shape)
pp=np.argmax(Y_test)
print('pp argmax shape:',pp.shape)
print(pp)
print(y_test[1:5])


ovr = model.predict(X_test)
overall = np.argmax(ovr,axis = 1)
print ('predict classes shape:',overall.shape)
print(overall)



target_names = ['class 0(a0)', 'class 1(a1)', 'class 2(a2)','class 3(a3)','class 4(a4)', 'class 5(a5)', 'class 6(a6)','class 7(a7)'
                ,'class 8(a8)', 'class 9(a9)','class 10(a10)', 'class 11(a11)', 'class 12(a12)','class 13(a13)','class 14(a14)', 'class 15(a15)', 'class 16(a16)']
					

#print('Prediction Y:',Y_pred.shape)
#print('prediction after argmax:', y_pred.shape)
print('shape of y test:',y_test.shape)
print(y_test)

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


indices = []

for i in range(0,72):
    array = []
    array = Y_test[i]
    #indices.append(array.index(1))
    occ = 0
    for j in array:
        occ +=1
        if j == 1:
            break
    indices.append(occ-1)



# Compute confusion matrix
cnf_matrix = (confusion_matrix(y_test, overall))

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
