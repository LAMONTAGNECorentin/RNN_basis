import numpy as np
import math
import matplotlib.pyplot as plt

#Split sequence function to create training batch
def split_sequence(sequence, split_size):
    X, Y = list(), list()
    for i in range(len(sequence)-split_size):
        x, y = sequence[i:(i+split_size)], sequence[i+split_size]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)                   #return as numpy array

#Create linear serie
def serie(start, end, step=1):
    dataset = []
    for i in np.arange(start, end, step):
        dataset.append(i)
    return dataset

def sinus(start, end, step, amplitude=1, style=0):
    dataset = []
    for i in np.arange(start, end, step):
        x = math.sin(i)*amplitude + i*style
        dataset.append(x)
    return dataset

def show(dataset):
    plt.figure(figsize=(12, 6))
    plt.plot(dataset, label='dataset')
    plt.title('Visualization')
    plt.xlabel('')
    plt.ylabel('')
    plt.legend()
    plt.show()