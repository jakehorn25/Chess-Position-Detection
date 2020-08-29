import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

from matplotlib.colors import NoNorm
import random
import os

from skimage import io, color

from tensorflow.keras.models import load_model

channels = 3
gray = False
if gray: 
    channels = 1

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
        X.append(io.imread('img/'+folder+e, as_gray=gray))
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

def vectorizeFEN(lst, length):
    acc = []
    for fen in lst:
        vector = [char for char in fen]
        while len(vector) < length:
            vector.append(' ')
        acc.append(np.array(vector))
    return acc

def displayMatrix(matrix):
    pass

def to64squares(img):
    size = int(400/8)
    squares = []
    i,j = 0,0
    for _ in range(64):
        squares.append(img[j*size:(j+1)*size,i*size:(i+1)*size]) ###
        i,j = nextSquare(i,j)
    
    return np.array(squares)

def plotSquares(squares):
   
    fig, axs = plt.subplots(8,8)
    fig.patch.set_facecolor('black')
    plt.rcParams['savefig.facecolor']='black'
    for i, ax in enumerate(axs.flatten()):
        if gray: ax.imshow(squares[i], cmap='gray', norm=NoNorm())
        else :   ax.imshow(squares[i])
        ax.set_xticks([])
        ax.set_yticks([])
        
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
    return Xs.reshape(-1,50,50,channels),ys

def getWeights(ys):
    weights = ys.sum(axis=0).max()/ys.sum(axis=0)
    
    return weights.reset_index().iloc[:,1].to_dict()

def plotWeights(ys):
    weights = ys.sum(axis=0).max()/ys.sum(axis=0)
    weights.plot.bar()
    plt.show()
    pass

def yhatToMatrix(yhat, columns):
    frame = pd.DataFrame(yhat, columns=columns)
    pieces = frame.idxmax(axis=1)
    arr = np.array(pieces).reshape(-1,8,8)
    return arr

def getErrorIndicies(yhat,y):
    yhm = yhatToMatrix(yhat, y.columns)
    ym =  yhatToMatrix(y, y.columns)
    idx = np.where(yhm!=ym)
    return idx

def plotBoard(idx, X, y):
    X = X.reshape(-1,64,50,50,channels)
    for i in range(len(idx)):
        board = X[idx[i],:,:,:]
        board = board.reshape(64,50,50,channels)

        img= reshape_squares(board,8,8)
        
        fig, axs = plt.subplots(1,2, figsize = [10,5])
        fig.patch.set_facecolor('black')
        plt.rcParams['savefig.facecolor']='black'
        fig.patch.set_alpha(.5)
        if gray: axs[0].imshow(img.reshape(img.shape[0],-1), cmap='gray', norm=NoNorm())
        else:    axs[0].imshow(img)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].text(0,0, matrixToText(y[idx[i]]), 
                    color='white',
                    fontproperties='monospace',
                    fontsize=25)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_frame_on(False)

        plt.tight_layout()
        plt.show()
    pass

def matrixToText(y):
    string = ''
    for arr in y:
        string += '|'
        for e in arr:
            string += e + '|'
        string += '\n'
    return string

def plotErrors(model, testX, testy):
    yhat = model.predict(testX)
    yhm = yhatToMatrix(yhat, testy.columns)
    
    errors = getErrorIndicies(yhat,testy)
    
    if errors:
        plotBoard(errors[0],testX,yhm)
    pass

def reshape_squares(imgs,nrow,ncols):
    
    if nrow*ncols == len(imgs):
        squares = np.vsplit(imgs, len(imgs))
        row = np.concatenate(squares, axis=2)
        temp = row.reshape(50,-1, channels)
        rows = np.split(temp, nrow, axis=1)
        return np.concatenate(rows, axis=0)
    else:
        print('row/col mismatch')
        return None
    pass

def we30Kings(n=30):
    X,y = importXy(n=n)
    kings = y.index[y['k']==1]
    kimg = X[kings]
    
    img= reshape_squares(kimg[:30,:,:,:],3,10)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    plt.rcParams['savefig.facecolor']='black'
    fig.patch.set_alpha(.5)
    if gray: ax.imshow(img.reshape(img.shape[0],-1), cmap='gray', norm=NoNorm())
    else:    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
        
    we30Kings()
    
    '''
    X,y = importXy('test/', n=100)
    model = load_model('models/final.h5')
    plotErrors(model, X, y)
    '''
    '''
    X,y = importImages(n=1)
    squares = to64squares(X[0])
    plotSquares(squares)
    '''
    #ys = yToSquares(y)
    
    #squares = to64squares(X[0])
    #plotSquares(Xs)
    #plt.show()
    