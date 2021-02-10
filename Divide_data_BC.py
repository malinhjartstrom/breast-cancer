# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:55:22 2021

@author: Malin
"""
# imports for importing and writing data
import pandas as pd
import xlwt
import numpy as np
    
# import for dividing the data
from sklearn.model_selection import train_test_split
# %% importing data using pandas
riskData = pd.read_excel('C:/Users/Malin/Documents/LTH/Risk_factors_copy.xls');
print(riskData)

# %% Imputations for missing data
columns_to_impute_999 = ['Menopausal_status', 'BMI', 'Tumour_size', 'Ki67_percentage']
columns_to_impute_99 = ['Multifocality']
columns_to_impute_9 = ['Histological_Grade', 'Lymphovascular_Invasion', 'ERstatus', 'PRstatus']

def imputation(columns_to_impute, missing_value):
    impute = riskData[columns_to_impute]
    for col in list(impute.columns):
        column = impute[col].values
        mn = np.min(column)
        nine = 0
        for row in range(len(column)):
            if column[row] == missing_value:
                nine = nine + 1
        s = pd.Series(column)
        largest = s.nlargest(n = nine + 1, keep='last')
        mx = min(largest)
        impute[col] = impute[col].replace(missing_value, np.random.randint(mn,mx+1))
    riskData[columns_to_impute] = impute
    
imputation(columns_to_impute_999, 999)
imputation(columns_to_impute_99, 99)
imputation(columns_to_impute_9, 9)
riskData['HER2status'] = riskData['HER2status'].fillna(np.random.randint(2));

# Make sure all missing values are taken care of
missing = 0
for col in range(2, len(riskData.columns)): # skipping rows for "fortnr" and "löpnr"
    for row in range(len(riskData)):
        value = riskData.iloc[row, col]
        if value == 99 or value == 999: # checking for missing values coded with 99 or 999
            missing = missing + 1
missing = missing + riskData.isnull().sum().sum() # checking for nan values
if missing != 0:
    print("There are still " + str(missing) + " missing values.")
else:
    print("All missing values have been taken care of (OBS: Have not checked for 9s).")
        
    
# %% Normalize continuous data [0,1]
columns_to_normalize = ['Age', 'BMI', 'Ki67_percentage', 'Tumour_size']
x = riskData[columns_to_normalize].values # extracting values
temp = np.zeros(x.shape)
for col in range(4):
    mx = np.max(x[:,col])
    mn = np.min(x[:,col])
    temp[:,col] = (x[:,col] - mn)/(mx-mn) # normalizing

riskData[columns_to_normalize] = temp # adding the normalized values

# %% Dividing into test vs training+validation data sets
labels = pd.concat([riskData['ID (Fortnr)'], riskData['Löpnr'], riskData['N0'], riskData['N2+']], axis=1)
data = riskData.drop(['ID (Fortnr)', 'Löpnr', 'N0', 'N2+'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8, random_state=42)


# %% Concatenate labels and data for train and test respectvely
riskTrainVal = pd.concat([y_train, x_train], axis=1)
riskTest = pd.concat([y_test, x_test], axis=1)

# %% Creating excel files of the test dataset and the training+validation dataset
path = 'C:/Users/Malin/Documents/LTH/'
fileName1 = 'risk_train_data'
xlsWriter = pd.ExcelWriter(str(path + fileName1 + '.xls'), engine = 'xlwt')

# Convert the dataframe to an Excel Writer object.
riskTrainVal.to_excel(xlsWriter)

# Close the Pandas Excel writer and output the Excel file.
xlsWriter.save()
xlsWriter.close()

fileName2 = 'risk_test_data'
xlsWriter = pd.ExcelWriter(str(path + fileName2 + '.xls'), engine = 'xlwt')

# Convert the dataframe to an Excel Writer object.
riskTest.to_excel(xlsWriter)

# Close the Pandas Excel writer and output the Excel file.
xlsWriter.save()
xlsWriter.close()

