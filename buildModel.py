#regression model in keras

import numpy as np
from keras.layers import Dense
from keras.models import Sequential

pred = np.loadtxt('predictors_data.csv',delimiter = ',')
cols = pred.shape[1]

#building
model = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (cols,)))	#input layer
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1))	#output layer

#compiling
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#fitting
model.fit(pred, target)