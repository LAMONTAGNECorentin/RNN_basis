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
import torch.optim as optim
import model
import Myfunction

torch.manual_seed(100)

dataset = dc.sinus(start=0, end=100, step=0.1, amplitude=5, style=0)
dataset_test = dc.sinus(start=0.1, end=100.1, step=0.1, amplitude=5, style=0)

f16_ref, leopard_ref, volvo_ref, destroyerengine_ref = Myfunction.read_noise('')
f16_ref_temp, leopard_ref_temp, volvo_ref_temp, destroyerengine_ref_temp = np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1])
# アップサンプリング実行を一時保存用配列へ
f16_ref_temp = Myfunction.UPSAMPLING(signal=f16_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)

NOISE_train_ref = np.vstack([f16_ref[:divide], leopard_ref[:divide], volvo_ref[:divide], destroyerengine_ref[:divide]])
NOISE_train_pri = np.vstack([f16_pri[:divide], leopard_pri[:divide], volvo_pri[:divide], destroyerengine_pri[:divide]])
# ################################################################################################
# (4, 5203786, 1)
NOISE_test_ref = np.stack([f16_ref[divide:], leopard_ref[divide:], volvo_ref[divide:], destroyerengine_ref[divide:]], axis=0)
NOISE_test_pri = np.stack([f16_pri[divide:], leopard_pri[divide:], volvo_pri[divide:], destroyerengine_pri[divide:]], axis=0)

split_size = 30
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

time_steps = np.arange(split_size, len(dataset_test))

plt.ion()

figure, ax = plt.subplots(figsize=(8, 6))
(line1,) = ax.plot(time_steps, testY)
plt.plot(time_steps, testY, label='Original Data')

#-------------------------------HYPERPARAMETERS-------------------------------------------
lr=0.005
input_size=1
hidden_size= 16
num_layers=1
output_size=1
num_epochs = 1000
h0, c0 = None, None
#----------------------------------------------------------------------------------------

model = model.LSTM_model(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

Loss_history = []
prediction = []

for epoch in range(num_epochs):
    model.train()
    #add scheduler here
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

