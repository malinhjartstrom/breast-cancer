# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:42:03 2021

@author: Malin
"""
# imports for importing data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# imports for deep learning
import tensorflow as tf
import tensorflow.keras
import sklearn
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling1D,UpSampling3D, AveragePooling1D#, Conv1DTranspose
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# %% importing data using pandas
dataFrame = pd.read_excel('C:/Users/Malin/Documents/LTH/Master_thesis/Lund_Cohort_LD_2009_2012_Database_Mammography_Node_Prediction_Magnus_Dustler_20201124_original.xlsx');
print(dataFrame)
# %% Create short data set for experimenting
dataFrameExp = dataFrame[0:10] #dataFrame.loc[:10]
print(dataFrameExp)

#drop columns
dataFrameExp = dataFrameExp.drop(['FortNr', 'Löpnr', 'Ankomstdat', 'year', 'Overall_Nodal_Status_Endpoints_Main', 'Overall_Nodal_Status_N0_N1_N2', 'Sentinel_Node_Status_Endpoint', 'Sentinel_Node_Status_N0_Nmicro_Nmacro', 'Clinical_Variables', 'Tumour_Pathological_Variables', 'Number_of_carcinomas', 'Transpara_score', 'filter_$'], axis=1)
print(dataFrameExp)
# Still missing Menopause column and should change columns of klockslag/breast

# %% Handle missing values for labels
dataFrameExp['Sentinel_Node_Status_N0_Npos'] = dataFrameExp['Sentinel_Node_Status_N0_Npos'].fillna(np.random.randint(2))

# %% Create labels
#y = dataFrameExp['Overall_Nodal_Status_N0_Npos', 'Sentinel_Node_Status_N0_Npos']
y1 = dataFrameExp['Sentinel_Node_Status_N0_Npos']
y2 = dataFrameExp['Overall_Nodal_Status_N0_Npos']
print(y1)
print(y2)

# %% Create inputs
x = dataFrameExp.drop(['Overall_Nodal_Status_N0_Npos', 'Sentinel_Node_Status_N0_Npos'], axis=1)
print(x)

# %% Handle missing values for inputs
x['BMI'] = x['BMI'].replace(999, np.random.randint(30))
x['Age'] = x['Age'].replace(999, np.random.randint(100));
x['Tumour_Size'] = x['Tumour_Size'].replace(999.00, np.mean(x.Tumour_Size));
x['HER2status'] = x['HER2status'].fillna(np.random.randint(2));
x['Ki67_percentage'] = x['Ki67_percentage'].replace(999, np.random.randint(101));

# %% Divide in training and test data set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y1, test_size=0.2, random_state=42);
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42);

# %%
num_features = x.shape[1]
print("We have " + str(num_features) + " features.");

# %% Convert to np
ytrain_np = y_train.to_numpy()
xtrain_np = X_train.to_numpy()
yval_np = y_val.to_numpy()
xval_np = X_val.to_numpy()
ytest_np = y_test.to_numpy()
xtest_np = X_test.to_numpy()
# print(xtrain_np.shape)

# %% Making output categorical???
ytrain_cat = tensorflow.keras.utils.to_categorical(ytrain_np)
yval_cat = tensorflow.keras.utils.to_categorical(yval_np)
ytest_cat = tensorflow.keras.utils.to_categorical(ytest_np)

# %%
#Create model
input_shape = Input(shape = (num_features), dtype='float32', name='main_input')

layer = Dense(15, activation='relu')(input_shape)
layer = Dense(1, activation='sigmoid')(layer)

model = Model(input_shape,
                    layer)

    #Compile model
model.compile(loss='binary_crossentropy',
                    optimizer = Adam(0.01), 
                    metrics=['accuracy']
                   )
    
    #Show model
model.summary()

    #Fit model
history = model.fit(xtrain_np, ytrain_np, 
                                    #batch_size=10,
                                    epochs=10,
                                    validation_data=(xval_np, yval_np),
                                    verbose=1,
                                   )

prediction = model.predict(xval_np)
prediction_cat = tensorflow.keras.utils.to_categorical(prediction);

# %% Build a confusion table
import seaborn as sn
from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction_cat,yval_np))
conMat = confusion_matrix(prediction_cat, 
                            yval_np, 
                            labels=None, 
                            sample_weight=None
                           )
sn.heatmap(conMat, annot=True)

# %% ROC & AUC
from sklearn import metrics
# calculting ROC score
#metrics.roc_curve(yval_np, prediction_cat)
# calculating AUC score
metrics.roc_auc_score(yval_np, prediction_cat)
# plot ROC curve
metrics.plot_roc_curve(model, prediction_cat, yval_np)
plt.show()  


