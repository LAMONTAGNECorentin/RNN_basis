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
def linear(start, end, step=1):
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

def sinus_nonlinear(start=0, end=10, step=0.01, amplitude=1, style=0):
    t = np.arange(start, end, step)
    sinus = amplitude * np.sin(2 * np.pi * t)
    
    if style == 1:
        return sinus + 0.3 * sinus**3                       # Distorsion non-linéaire cubique
    elif style == 2:
        return np.tanh(2 * sinus)                           # Fonction de saturation
    elif style == 3:
        return np.clip(sinus, -amplitude/2, amplitude/2)                    # Ecrêtage
    elif style == 4:
        modulator = 1 + 0.5 * np.sin(0.1 * 2 * np.pi * t)   # Modulation non-linéaire
        return modulator * np.sin(2 * np.pi * t)
    else:
        return sinus
    
