# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:21:06 2021

@author: hjart
"""


import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from PIL import Image
import sklearn 
import imageio
import seaborn as sn
import random as rn
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling1D, MaxPooling2D, UpSampling3D, AveragePooling2D#, Conv1DTranspose
from tensorflow.keras.models import Model ,Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.applications import Xception, VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# %% Import images
path = '/scratch/bob/malin_hj/decImages/revImages/' #Bob

import glob
import os

fortnrs = []
sides = []
dates = []
mammograms = []

for img_filename in os.listdir(path):
    if img_filename.endswith(".tif"):
      if "rev" in img_filename:
        fortnr, side, date, rev = img_filename.split('.')[0].split('_')
        fortnrs.append(int(fortnr))
        sides.append(side)
        dates.append(int(date))
        image = imageio.imread(path + img_filename)
        mammograms.append(image)        

fortnr = pd.DataFrame(fortnrs, columns=['Im_ID (Fortnr)'])
side = pd.DataFrame(sides, columns=['Im_Right_Left_Breast'])
date = pd.DataFrame(dates, columns=['Im_Date'])

imageInfo = pd.concat([fortnr, side, date], axis=1)
print('Image info: ' + str(imageInfo))


# %% Import excel file
riskFile = '/home/william/malin_hj/Documents/Breast_cancer/risk_train_data.xls' #training + validation data
riskData = pd.read_excel(riskFile)
riskData = riskData.drop(['Unnamed: 0'], axis=1)

# %% Combine mammograms with corresponding risk factor data
images = []
mammograms_riskData = pd.DataFrame()
no_img = 0

for fortnr in imageInfo['Im_ID (Fortnr)'].unique(): #every fortnr is examined one time
  possible_images_info = imageInfo[imageInfo['Im_ID (Fortnr)'].isin([fortnr])] #imageInfo of the possible images
  row = riskData.loc[riskData['ID (Fortnr)'] == fortnr]
  print('Fortnr: ' + str(fortnr))
  
  if row['Right_Left_Breast'].empty == False:
    breast = row['Right_Left_Breast']
    print('Breast value: ' + str(breast.values))
    latest_image_idx = -1
    image_info = -1
    
    #Find if to use left or right breast and then find the most reason image:
    if breast.values == 1:
      image_info = possible_images_info[possible_images_info['Im_Right_Left_Breast'].str.match('RCC')]
      if image_info['Im_Right_Left_Breast'].empty:
        no_img = no_img + 1
        continue # stop the iteration of the current "for" loop
      print('Right breast image info: ' + str(image_info))
    if breast.values == 2:
      image_info = possible_images_info[possible_images_info['Im_Right_Left_Breast'].str.match('LCC')]  
      print('Left breast image info: ' + str(image_info))  
      if image_info['Im_Right_Left_Breast'].empty:
        no_img = no_img +1
        continue
    latest_image_idx = image_info['Im_Date'].idxmax(axis=0)   
    
    # Making sure non of the non relevant data is used (training+val vs test):
    if latest_image_idx == -1:
      print("Did not find right information row.")
    else:   
      images.append(mammograms[latest_image_idx]) #Adding the images in the same order as the info
      image_info_final = possible_images_info.loc[possible_images_info.index.values == latest_image_idx] #image info of the final image
      print('Final info: ' + str(image_info_final))
      
      #Adding the image info to the riskdata info
      row['Im_ID (Fortnr)'] = image_info_final['Im_ID (Fortnr)'].values
      row['Im_Right_Left_Breast'] = image_info_final['Im_Right_Left_Breast'].values
      row['Im_Date'] = image_info_final['Im_Date'].values
      
      #Adding the new row of info to the resulting data frame
      mammograms_riskData = mammograms_riskData.append(row)
print('Fortnumbers without images: ' + str(no_img))
#print(mammograms_riskData)
#print(images)

# %% CNN
images = np.float32(images)
labels = mammograms_riskData['N0']
labels = np.array(labels)

# %% Xception pre-process
#xcep_images = tf.keras.applications.xception.preprocess_input(images)
#images= xcep_images
#print(images.shape)

# %% VGG16 pre-process
def makeRGBshape(images):
  images3channels = [] #to make input in the shape of RGB (600,400,3)
  for idx in range(len(images)):
    temp = [images[idx], images[idx], images[idx]]
    temp = np.array(temp)
    images3channels.append(temp)
  images3channels = np.array([images3channels])
  print('Shape of the 3 channel array: ' + str(images3channels.shape))
  return images3channels
  
#prepared_images = preprocess_input(images3channels)
#print(prepared_images.shape)
#images = prepared_images

# %%
#(train_images, train_labels),(val_images,val_labels) = '' #load train/validation data
image_train, image_val, label_train, label_val = train_test_split(images, labels, test_size=0.2, random_state=42);

image_train_rgb = makeRGBshape(image_train)
image_val_rgb = makeRGBshape(image_val)
print('Shape of image val rgb: ' + str(image_val_rgb.shape))

image_train_rgb = image_train_rgb.reshape(image_train_rgb.shape[1], image_train_rgb.shape[3],image_train_rgb.shape[4],3)
image_val_rgb = image_val_rgb.reshape(image_val_rgb.shape[1], image_val_rgb.shape[3], image_val_rgb.shape[4],3)

image_train_rgb = preprocess_input(image_train_rgb)
image_val_rgb = preprocess_input(image_val_rgb)
print('Shape of image val rgb efter Keras preprocessing: ' + str(image_val_rgb.shape))
print('Shape of training labels: ' + str(label_train.shape))
label_train_preprocessed = preprocess_input(label_train)
print('Shape of training labels after preprocessing: ' + str(label_train_preprocessed.shape))

width = image_train_rgb[0].shape[1]
print('Width: ' + str(width))
height = image_train_rgb[0].shape[0]
print('Height: ' + str(height))
#input_shape = (height, width, 1)
#image_train = image_train.reshape(image_train.shape[0],image_train.shape[1],image_train.shape[2],3)
#image_val = image_val.reshape(image_val.shape[0],image_val.shape[1],image_val.shape[2],3)

#print(input_shape)
#print(image_train.shape)
#print(image_train)


# %% Xception
#xceptionModel = tf.keras.applications.Xception(include_top = False)
##x = xceptionModel(input_shape)
#print(xceptionModel.output)
#print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#print(xceptionModel.input)

#layer = Conv2D(16,8, activation='relu', padding = 'same') (xceptionModel.output)
#flat = Flatten(layer)

# %% VGG16 model
new_input = Input(shape=(600, 400, 3))
print('Input shape:')
print(new_input)
model = VGG16(include_top = False, input_tensor = new_input)
#model.summary()
x = model(model.inputs, training=False)
x = MaxPooling2D()(x)
x = Flatten()(x)
finalOutput = Dense(1, activation = 'sigmoid')(x)
finalModel = Model(inputs=new_input, outputs=finalOutput)
finalModel.summary()

#model = Model(inputs=model.inputs, outputs=model.output)

#output=Dense(1, activation='sigmoid')(xceptionModel.output)
#model = Model(xceptionModel.input, output)

# %% Testing CNN

#model = Sequential()
#model.add(Conv2D(2, 9, activation='relu', padding='same', input_shape=input_shape))
#model.add(MaxPooling2D(pool_size = 2))
#model.summary()
#model.add(Conv2D(2,9, activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size = 2))
#model.add(Conv2D(1,3, activation = 'relu', padding='same'))
#model.add(MaxPooling2D(pool_size = 2))
#model.summary()
#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))

#Create model
#model = Model(input_shape, output)

#Compile model
model.compile(loss='binary_crossentropy',
                optimizer = Adam(0.01),
                metrics=['accuracy', 'AUC']
               )

#Show model
model.summary()

history = model.fit(image_train_rgb, label_train, 
                        batch_size=1,
                        epochs=10,
                        validation_data=(image_val_rgb, label_val),
                        verbose=1
                        #callbacks = my_callbacks,
                       )

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('binary_accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()


pred_trn = model.predict(image_train).reshape(label_train.shape)
pred_val = model.predict(image_val).reshape(label_val.shape)


# %% ROC & AUC
fpr, tpr, thresholds = metrics.roc_curve(label_val, pred_val, pos_label=1) # label = 1 is healthy for the NO vs N+ test
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='CNN model')
display.plot()
plt.title('ROC curve')
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity') 
plt.show()
# %%
#CNN model
# =============================================================================
# def cnnModel(input_shape, filters, ks,layers, p_size, p_type):
#     cnn = Sequential()
#     
#     print('*@*@*@*@*@*@')
#     print(input_shape)
#     cnn.add(Conv2D(filters[0],kernel_size = (ks[0],ks[0]), activation = 'relu', padding = 'same', strides = 1, input_shape = input_shape)) #Conv2D(filters, kernel_size, )
#     #print(cnn.shape)
#     if p_type == 1: #max pooling
#         cnn.add(MaxPooling2D(pool_size = (p_size[0],p_size[0])))
#     else: #p_type == 2
#         cnn.add(AveragePooling2D(pool_size = (p_size[0],p_size[0])))
#     
#     for i in range(layers-1):
#         cnn.add(Conv2D(filters[i+1],kernel_size = (ks[i+1],ks[i+1]), activation = 'relu', padding = 'same', strides = 1)) #Conv2D(filters, kernel_size, )
#         if p_type == 1: #max pooling
#             cnn.add(MaxPooling2D(pool_size = (p_size[i+1],p_size[i+1])))
#         else: #p_type == 2
#             cnn.add(AveragePooling2D(pool_size = (p_size[i+1],p_size[i+1])))
# 
#     #flatten layer
#     cnn.add(Flatten())
#     
#     print('*@*@*@*@*@*@')
#     #output layer - Do we need this when we just want our features as output?
#     cnn.add(Dense(1, activation = 'sigmoid'))
#     return cnn
# 
# 
# #cnn = cnnModel(input_shape, np.array(([1,1,1])), np.array((3,3,3)))
# 
# def cnnCompile(cnn,lr):
#     print('@@@@@@@@@@@@')
#     cnn.compile(loss='binary_crossentropy',
#     optimizer = Adam(lr), #Adam(learning rate)
#     metrics=['accuracy'])
# 
#     #Show model
#     cnn.summary()
# 
#     history = cnn.fit(image_train, label_train, 
#                             batch_size=10,
#                             epochs=100,
#                             validation_data=(image_val, label_val),
#                             verbose=0
#                             #callbacks = my_callbacks,
#                            )
#     print(history.history.keys())
#     # summarize history for accuracy
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
#     plt.ylabel('binary_accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['Train', 'Val'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['Train', 'Val'], loc='upper left')
#     plt.show()
#  # %%
# def print_variables(lr,bs,reg,drop,k,f,layers,p_size, p_type, input_shape): #print the variables
#      print('Leraning rate random: ')
#      print(lr)
#      print('Batch size random:')
#      print(bs)
#      print('Regularization random:')
#      print(reg)
#      print('Dropout random:')
#      print(drop)
#      print('Kernels for each layer random')
#      print(k)
#      print('Filters for each layer random:')
#      print(f)
#      print('Number of layers random:')
#      print(layers)
#      print('Poolin size random:')
#      print(p_size)
#      print('Pooling type random:')
#      if p_type == 1:
#          print('Max pooling')
#      else:
#          print('Average pooling')
#      print('Input shape: ')
#      print(input_shape)
#  
#  # %%
# #random selection of hyperparameters to feed into model:
# def cnnHyperparTune(lr, bs, reg, drop, k, f, input_shape, layers, p): 
#     #Random search
#     k_comb = []
#     f_comb = []
#     p_comb = []
#     for indx in range(layers):
#         k_comb.append(k[indx][rn.randint(0,len(k[indx]))-1])
#         f_comb.append(f[indx][rn.randint(0,len(f[indx]))-1])
#         p_comb.append(p[indx][rn.randint(0,len(p[indx]))-1])
# 
#     lr_rand = lr[rn.randint(0, len(lr)-1)]
#     bs_rand = bs[rn.randint(0, len(bs)-1)]
#     reg_rand = reg[rn.randint(0, len(reg)-1)]
#     drop_rand = drop[rn.randint(0, len(drop)-1)]
#     layers_rand = rn.randint(1,layers) #minimum one layer
#     p_type_rand = rn.randint(1, 2) #1 equals max pooling, 2 equals average pooling
#     
#     print_variables(lr_rand, bs_rand, reg_rand, drop_rand, k_comb, f_comb, layers_rand, p_comb, p_type_rand, input_shape)
#     
#     cnn = cnnModel(input_shape, f_comb, k_comb,layers_rand, p_comb, p_type_rand) #feed hyperparameters into CNN model
#     cnnCompile(cnn,lr_rand) #compile CNN model 
#                     
#     return cnn
# 
# # %% MAIN
# lr = np.logspace(-6, -2, num=5) #options for learning rate
# bs = [32, 64, 128] #options for batch size
# reg = ['l1', 'l2'] #options for regularization
# drop = [0, 0.25, 0.5] #options for dropout 
# kernel_sizes = [[2,3,4],[2,3,4],[2,3],[2,3,4,5],[2,3,4,5]] #possible kernel sizes in different layers
# filters = [[1],[1],[1],[1],[1]] #posible filter sizes in different layers
# pool_size = [[2,3],[2,3],[2,3],[2,3],[2,3]] #possible pool sizes for different layers
# #pool_type = [] #max pooling or average pooling
# layers = 5 #layers including first convolutional layer with images as input
# cnn = cnnHyperparTune(lr, bs, reg, drop, kernel_sizes, filters, input_shape,layers,pool_size) #run method that randomly selects hyperparameters
# 
# 
# # %%
# pred_trn = cnn.predict(image_train).reshape(label_train.shape)
# pred_val = cnn.predict(image_val).reshape(label_val.shape)
# 
# 
# # %% ROC & AUC
# fpr, tpr, thresholds = metrics.roc_curve(label_val, pred_val, pos_label=1) # label = 1 is healthy for the NO vs N+ test
# roc_auc = metrics.auc(fpr, tpr)
# display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='CNN model')
# display.plot()
# plt.title('ROC curve')
# plt.xlabel('1 - Specificity')
# plt.ylabel('Sensitivity')
# plt.show()
# 
# =============================================================================
  