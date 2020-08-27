from functions import importXy, getWeights, plotErrors

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(0)

def define_model(filters, kernel_size, shape, classes):
    model = Sequential()
    activators = ['relu']#'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    model.add(Conv2D(filters, kernel_size, padding = 'valid', input_shape = shape))
    model.add(Activation(activators[0]))

    #last layer
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    return model

if __name__ == '__main__':
    
    classes = 13
    filters = 20 
    kernel_size = (5,5)
    shape = (50,50,1)

    model= define_model(filters, kernel_size, shape, classes)
    rounds = 20
    for i in range(rounds):
        print (f'Round {i+1} of {rounds}')
        Xtr ,ytr = importXy('train/', 80)
        Xte, yte = importXy('train/' , 20)
        
        model.fit(
            x=Xtr, y=ytr, batch_size=64, epochs=1, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=(Xte,yte), shuffle=False, class_weight=getWeights(ytr),
            sample_weight=None, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=True)

    Xtr ,ytr = 0,0
    Xte, yte = importXy('test/' , 100)
    score = model.evaluate(Xte,yte, verbose=0)
    print('Test score:',score[0])
    print('Test accuarcy:',score[1])

    plotErrors(model, Xte, yte)
    file = 'throwaway'
    breakpoint()
    model.save(f'models/{file}.h5')
