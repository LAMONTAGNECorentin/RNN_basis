#=================================================================================================
# Programer :       Corentin LAMONTAGNE
# Starting date :   2025/06/02
#- - - - - - - - - - - - - - - - - - - - -  OBJECTIVES - - - - - - - - - - - - - - - - - - - - - -
#   o   Make a functionnal LSTM for simple dataset
#   o   Learn basis and improve uderstanding of hyperparameters effects.
#   o   Improve torch library knowledge
#- - - - - - - - - - - - - - - - - - - - - - SOURCES - - - - - - - - - - - - - - - - - - - - - - -
# Web : https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# Web : https://www.geeksforgeeks.org/long-short-term-memory-networks-using-pytorch/
#=================================================================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math


#Split sequence function to create training batch
def split_sequence(sequence, split_size):
    X, Y = list(), list()
    for i in range(len(sequence)-split_size):
        x, y = sequence[i:(i+split_size)], sequence[i+split_size]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)                   #return as numpy array

dataset = []
for i in range(0,100):
    dataset.append(i)

dataset_test = []
for i in range(10,110):
    dataset_test.append(i)
    
print("init")

split_size = 5
X, Y = split_sequence(dataset, split_size)
Xt, Yt = split_sequence(dataset_test, split_size)
# for i in range(len(X)):
# 	print(X[i], Y[i])

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(Y[:, None], dtype=torch.float32)

trainXt = torch.tensor(Xt[:, :, None], dtype=torch.float32)
trainYt = torch.tensor(Yt[:, None], dtype=torch.float32)

class LSTM_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)   #Initialisation h0=0
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)   #Initialisation c0=0
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn

#----HYPERPARAMETERS----
lr=0.005
input_dim=1
hidden_dim= 128
layer_dim=1
output_dim=1
num_epochs = 2000
h0, c0 = None, None
#----------------------

model = LSTM_model(input_dim, hidden_dim, layer_dim, output_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

Loss_history = []
prediction = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs, h0, c0 = model(trainX, h0, c0)

    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    h0 = h0.detach()
    c0 = c0.detach()

    Loss_history.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
    if (epoch+1) % int(num_epochs/4) == 0:
        model.eval()
        with torch.no_grad():
            predicted, _, _ = model(trainXt, h0, c0)
            prediction.append(predicted.detach().numpy())


torch.save(model.state_dict(), 'test')
        
model.eval()
predicted, _, _ = model(trainXt, h0, c0)

original = dataset_test[split_size:]
time_steps = np.arange(split_size, len(dataset_test))

# plt.figure(figsize=(12, 6))

# plt.subplot(1,4,1)
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, prediction[-4], label='Predicted Data', linestyle='--')
# plt.subplot(1,4,2)
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, prediction[-3], label='Predicted Data', linestyle='--')
# plt.subplot(1,4,3)
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, prediction[-2], label='Predicted Data', linestyle='--')
# plt.subplot(1,4,4)
# plt.plot(time_steps, original, label='Original Data')
# plt.plot(time_steps, prediction[-1], label='Predicted Data', linestyle='--')

# plt.title('LSTM Model Predictions vs. Original Data')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.show()


plt.figure(figsize=(12, 6))
plt.plot(Loss_history, label='Loss')
plt.title('LSTM Loss evolution')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(time_steps, original, label='Original Data')
plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()
