# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:55:42 2020

@author: Aparajita Das
"""
# Part 1 -Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(35)

# Importing the training set
dataset_train = pd.read_csv('indiafull.csv')
training_set = dataset_train.iloc[:, 3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure 
X_train = []
y_train = []
for i in range(85, 102):
    X_train.append(training_set_scaled[i-85:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 45, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer nd some Dropout regularisation
regressor.add(LSTM(units = 45, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 45, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 45))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 80 , batch_size = 20)



# Part 3 - Making the predictions 

# Getting the real data
dataset_test = pd.read_csv('indiatest.csv')
real_confirmed_rate = dataset_test.iloc[:, 3:4].values

# Getting the predicted data
dataset_total = pd.concat((dataset_train['Deaths'], dataset_test['Deaths']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 85:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(85, 95):
    X_test.append(inputs[i-85:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_confirmed_rate = regressor.predict(X_test)
predicted_confirmed_rate = sc.inverse_transform(predicted_confirmed_rate)



# Part 4 - Visualising the results

# Making structure for visualising

df_old = pd.read_csv('indiafull.csv', usecols = ['Date', 'Deaths'])
df_pred = pd.read_csv('indiatest.csv', usecols = ['Date'])
df_pred['Deaths'] = predicted_confirmed_rate
frames = [df_old, df_pred]
df_result = pd.concat(frames)

copy = df_result
copy = copy.drop('Date', axis=1)
copy_df_date = df_result
copy_df_date = copy_df_date.drop('Deaths', axis=1)

# Visualizing predicted Data
datelist2 = list(copy_df_date.iloc[:, 0].values)
copy['Date'] = datelist2 
copy = copy.set_index(['Date'])
copy.plot()


 
dates = list(dataset_test.iloc[:, 0].values)
df_3 = pd.DataFrame(predicted_confirmed_rate)
df_4 = dates
df_3['Date'] = df_4
df_3 = df_3.set_index(['Date'])
df_54 = copy[:102].copy(deep = True)
df_54.plot()

#visualization of future forecast/prediction
plt.plot(df_54, color = 'blue', label = 'Real Covid19 Death Case')
plt.plot(copy, color = 'red', label = 'Predicted Covid19 Deaths Case', alpha = 0.4)
plt.title('India Covid19 Daywise Deaths Prediction')
plt.xticks(rotation=60)
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(3))
plt.tight_layout()
plt.xlabel('Days')
plt.ylabel('Cases')
plt.legend()
plt.show()