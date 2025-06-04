import numpy as np

#Split sequence function to create training batch
def split_sequence(sequence, split_size):
    X, Y = list(), list()
    for i in range(len(sequence)-split_size):
        x, y = sequence[i:(i+split_size)], sequence[i+split_size]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)                   #return as numpy array

def serie(start, end, step=1):
    dataset = []
    for i in range(start, end, step):
        dataset.append(i)
    return dataset