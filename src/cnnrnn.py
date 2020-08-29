from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from functions import importImages, vectorizeFEN

def define_model(filters, kernelsize, input_shape, poolsize, units=64):
    
    activators = ['linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign']

    model = Sequential()

    #400,400
    model.add(Conv2D(filters, kernelsize, padding='valid', input_shape=input_shape))
    model.add(Activation(activators[3]))
    model.add(Conv2D(filters, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(MaxPooling2D(pool_size=poolsize))

    #200,200
    model.add(Conv2D(filters*2, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(Conv2D(filters*2, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(MaxPooling2D(pool_size=poolsize))

    #100,100
    model.add(Conv2D(filters*4, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(Conv2D(filters*4, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(MaxPooling2D(pool_size=poolsize))

    #50,50
    model.add(Conv2D(filters*8, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(Conv2D(filters*8, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(MaxPooling2D(pool_size=poolsize))

    #25,25
    
    model.add(Conv2D(filters*16, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    model.add(Conv2D(filters*16, kernelsize, padding='valid'))
    model.add(Activation(activators[3]))
    #model.add(MaxPooling2D(pool_size=poolsize))
    
    model.add(Flatten())
    #model.add(Dense(64*13))
    #model.add(Activation(activators[0]))
    
    model.add(Embedding(len(chars),units))
    model.add(LSTM(units, return_sequences=True))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='categorical_accuracy')

    return model

if __name__ == '__main__':
    chars = '12345678-bknpqrBKNPQR'
    X,y = importImages(n=20)

    nbfilters = 10
    kernelsize = (3,3)
    input_shape = (400,400,3)
    poolsize = (2,2)
    units = 64

    vy = vectorizeFEN(y, units)
    

    model = define_model(nbfilters, kernelsize, input_shape, poolsize, units)

    model.fit(np.array(X),np.array(vy), batch_size=5, epochs=4)