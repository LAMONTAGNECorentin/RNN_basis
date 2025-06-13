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
import dataset_creator as dc
import csv

torch.manual_seed(0)

# dataset = dc.linear(0,500,1)
# dataset_test = dc.linear(10,510,1)

dataset = dc.sinus(start=0, end=100, step=0.1, amplitude=5, style=1)
dataset_test = dc.sinus(start=0.1, end=100.1, step=0.1, amplitude=5, style=1)
# print(dataset_test[-1])
# dataset_test[-1]=-1

split_size = 10
X, Y = dc.split_sequence(dataset, split_size)
Xt, Yt = dc.split_sequence(dataset_test, split_size)
# for i in range(len(X)):
# 	print(X[i], Y[i])

trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(Y[:, None], dtype=torch.float32)
print(trainX.size())

testX = torch.tensor(Xt[:, :, None], dtype=torch.float32)
testY = torch.tensor(Yt[:, None], dtype=torch.float32)
print(testX.size())

class LSTM_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True, dropout=0.3, bidirectional=False, proj_size=0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)   #Initialisation h0=0
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)   #Initialisation c0=0
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out, hn, cn

time_steps = np.arange(split_size, len(dataset_test))

plt.ion()

figure, ax = plt.subplots(figsize=(8, 6))
(line1,) = ax.plot(time_steps, testY)
plt.plot(time_steps, testY, label='Original Data')

#-------------------------------HYPERPARAMETERS-------------------------------------------
lr=0.01
input_size=1
hidden_size= 16
num_layers=1
output_size=1
num_epochs = 1000
h0, c0 = None, None
#----------------------------------------------------------------------------------------

model = LSTM_model(input_size, hidden_size, num_layers, output_size)
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

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        model.eval()
        predicted, _, _ = model(testX, h0, c0)

        plt.title(f'Epoch [{epoch+1}/{num_epochs}]')
        line1.set_xdata(time_steps)
        line1.set_ydata(predicted.detach().numpy())
        figure.canvas.draw()
        figure.canvas.flush_events()
        

    if (epoch+1) % int(num_epochs/4) == 0:
        model.eval()
        with torch.no_grad():
            predicted, _, _ = model(testX, h0, c0)
            prediction.append(predicted.detach().numpy())


torch.save(model.state_dict(), 'test')

plt.ioff()

model.eval()
predicted, _, _ = model(testX, h0, c0)

with open('prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(predicted.detach().numpy())

original = dataset_test[split_size:]

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(Loss_history, label='Loss')
plt.title('LSTM Loss evolution')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()

plt.subplot(1,2,2)
plt.plot(time_steps, testY, label='Original Data')
plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
plt.title('LSTM Model Predictions vs. Original Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.show()

