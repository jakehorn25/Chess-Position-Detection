import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

from matplotlib.colors import NoNorm
import random
import os

from skimage import io, color

from tensorflow.keras.models import load_model

def importImages(string= 'train/', n=100):
    path = 'img/'+string
    files = os.listdir(path)
    subset = random.choices(files,k=n)
    X = loadImages(subset, string)
    y = getFENsfromSet(subset)
    return X,y

def getFENsfromSet(subset):
    for i, s in enumerate(subset):
        subset[i] = s[:-5]
    return subset

def loadImages(subset, folder):
    X = []
    for e in subset:
        X.append(color.rgb2gray(io.imread('img/'+folder+e)))
    return X

def FENtoMatrix(fen):
    matrix = np.empty([64],dtype='U')
    fen = fen.replace('-','')
    for i, _ in enumerate(matrix):
        if fen[0].isalpha():
            matrix[i] = fen[0]
            fen = fen[1:]
        else:
            matrix[i] = '_'
            if int(fen[0]) > 1:
                fen = str(int(fen[0])-1) + fen[1:]
            else: 
                fen = fen[1:]
    return matrix

def allFENtoCat(y):
    
    mList = [FENtoMatrix(y[i]) for i in range(len(y))]
    m = np.vstack(mList)
    return matrixToCategories(m.reshape(-1))

def matrixToCategories(matrix):
    #pieces = '_prnbqkPRBQK'
    categories = pd.get_dummies(matrix)
    return categories


def matrixToFEN(matrix):
    pass

def displayMatrix(matrix):
    pass

def to64squares(img):
    size = int(400/8)
    squares = []
    i,j = 0,0
    for _ in range(64):
        squares.append(img[j*size:(j+1)*size-1,i*size:(i+1)*size-1])
        i,j = nextSquare(i,j)
    
    return np.array(squares)

def plotSquares(squares):
    #io.imshow(X[0])
    #plt.title(y[0])
    _, axs = plt.subplots(8,8)
    
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(squares[i], cmap='gray', norm=NoNorm())
        ax.set_xticks([])
        ax.set_yticks([])
        
    #plt.tight_layout()
    plt.gray()
    plt.show()
    pass

def nextSquare(i,j):
    i += 1
    if i > 7:
        j+=1
        i=0 
    return i,j

def allToSquares(X):
    s = [to64squares(X[i]) for i  in range(len(X))]
    imgs = np.vstack(s)
    return imgs

def importXy(string= 'train/', n=100):
    X,y = importImages(string, n)
    Xs = allToSquares(X)
    ys = allFENtoCat(y)
    return Xs.reshape(-1,49,49,1),ys

def getWeights(ys):
    weights = ys.sum(axis=0).max()/ys.sum(axis=0)
    
    return weights.reset_index().iloc[:,1].to_dict()

def yhatToMatrix(yhat, columns):
    frame = pd.DataFrame(yhat, columns=columns)
    arr = np.array(frame.idxmax(axis=1)).reshape(-1,8,8)
    return arr

def getErrorIndicies(yhat,y):
    yhm = yhatToMatrix(yhat, y.columns)
    ym =  yhatToMatrix(y, y.columns)
    idx = np.where(yhm!=ym)
    return idx

def getBoard(idx, X):
    X = X.reshape(-1,64,49,49)
    board = X[idx[0],:,:,:]
    return board.reshape(64,49,49)

if __name__ == '__main__':
    #print (FENtoMatrix('1B1B2K1-1B6-5N2-6k1-8-8-8-4nq2'))
    X,y = importXy('test/', n=100)
    model = load_model('models/squaremodel.h5')

    yhat = model.predict(X)
    yhm = yhatToMatrix(yhat, y.columns)
    ym =  yhatToMatrix(y, y.columns)

    errors = getErrorIndicies(yhat,y)
    print(yhm[errors[0],:,:])
    if errors:
        plotSquares(getBoard(errors[0],X))

    #ys = yToSquares(y)
    
    #squares = to64squares(X[0])
    #plotSquares(Xs)
    #plt.show()
    