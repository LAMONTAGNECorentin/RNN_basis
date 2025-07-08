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

# dataset = dc.sinus(start=0, end=100, step=0.1, amplitude=5, style=0)
# dataset_test = dc.sinus(start=0.1, end=100.1, step=0.1, amplitude=5, style=0)

dataset = dc.sinus_nonlinear(start=0, end=10, step=0.01, amplitude=5, style=3)
dataset_test = dc.sinus_nonlinear(start=0.01, end=10.01, step=0.01, amplitude=5, style=3)

#----------------------------------------------------------------------------------------------------------
output_size = 1
test_dataset_size = 10000
TRAIN_SET_NUM = 10
SEQUENCE_SIZE = 20
Ts_train = 0.001                # Training period
predict_sample_num = 1
batchsize = 1
fs_resample = 44.1              # Frequency

fs_resample_khz = fs_resample
fs_resample = fs_resample*10**3

Ts = float(1/(fs_resample))             # frequency to periode
train_dataset_size = int(Ts_train/Ts) 

f16_ref, leopard_ref, volvo_ref, destroyerengine_ref = Myfunction.read_noise('')
f16_ref, leopard_ref, volvo_ref, destroyerengine_ref = f16_ref/(2**15), leopard_ref/(2**15), volvo_ref/(2**15), destroyerengine_ref/(2**15)

kind = 'cubic'
fs_orig=19.98*10**3             # 19.98Khz
size = int(len(f16_ref) * fs_resample / fs_orig)

# 一時保存用配列
f16_ref_temp, leopard_ref_temp, volvo_ref_temp, destroyerengine_ref_temp = np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1])
# アップサンプリング実行を一時保存用配列へ
f16_ref_temp = Myfunction.UPSAMPLING(signal=f16_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)
leopard_ref_temp = Myfunction.UPSAMPLING(signal=leopard_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)
volvo_ref_temp = Myfunction.UPSAMPLING(signal=volvo_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)
destroyerengine_ref_temp = Myfunction.UPSAMPLING(signal=destroyerengine_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)
# この後使う配列へ代入
f16_ref, leopard_ref, volvo_ref, destroyerengine_ref = np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1])

# f16, leopard, volvo, destroyerengine =f16_temp, leopard_temp, volvo_temp, destroyerengine_temp
f16_ref, leopard_ref, volvo_ref, destroyerengine_ref = f16_ref_temp.copy(), leopard_ref_temp.copy(), volvo_ref_temp.copy(), destroyerengine_ref_temp.copy()
# 一次信号の変数に、参照信号の値を代入する(_pri = _ref)
f16_pri, leopard_pri, volvo_pri, destroyerengine_pri = f16_ref_temp.copy(), leopard_ref_temp.copy(), volvo_ref_temp.copy(), destroyerengine_ref_temp.copy()

divide = int(f16_pri.shape[0]*0.5)

NOISE_train_ref = np.vstack([f16_ref[:divide], leopard_ref[:divide], volvo_ref[:divide], destroyerengine_ref[:divide]])
NOISE_train_pri = np.vstack([f16_pri[:divide], leopard_pri[:divide], volvo_pri[:divide], destroyerengine_pri[:divide]])

# ################################################################################################
# (4, 5203786, 1)
NOISE_test_ref = np.stack([f16_ref[divide:], leopard_ref[divide:], volvo_ref[divide:], destroyerengine_ref[divide:]], axis=0)
NOISE_test_pri = np.stack([f16_pri[divide:], leopard_pri[divide:], volvo_pri[divide:], destroyerengine_pri[divide:]], axis=0)

r_test = np.random.randint(0,len(NOISE_test_ref[0])-test_dataset_size-SEQUENCE_SIZE)

test_list = np.arange(r_test,r_test+test_dataset_size, dtype = 'int')

# 学習用のサンプル区間についてのリストを作成
r_train_list = np.random.randint(0,len(NOISE_train_ref)-train_dataset_size-SEQUENCE_SIZE, TRAIN_SET_NUM)
rand_train_list = np.zeros([TRAIN_SET_NUM, train_dataset_size],dtype = 'int')
for i in range(TRAIN_SET_NUM):
    rand_train_list[i,:] = np.arange(r_train_list[i],r_train_list[i]+train_dataset_size, dtype = 'int')

# ランダムにtrain_dataset_aize個サンプルを取り出してデータセットを作る
train_ref = np.zeros([TRAIN_SET_NUM*train_dataset_size, SEQUENCE_SIZE])
target_train = np.zeros([TRAIN_SET_NUM*train_dataset_size, output_size])
# trainセットの作成
for train_set_num in range(TRAIN_SET_NUM):
    for i in range(train_dataset_size) :
        r = rand_train_list[train_set_num,i]
        train_ref[train_dataset_size*train_set_num + i] = NOISE_train_ref[r:r+SEQUENCE_SIZE,0] # train:[train_dataset_size, 20]######################################################### 入力はref
        target_train[train_dataset_size*train_set_num + i] =NOISE_train_pri[r+SEQUENCE_SIZE,0] #target_train:[train_dataset_size, 1]######################################################### 教師はpri

# テストセットの生成
test = np.zeros([4, test_dataset_size, SEQUENCE_SIZE])
target_test = np.zeros([4, test_dataset_size, output_size])
# test_for_Einput_to_ref = np.zeros([4, test_dataset_size, output_size]) # Einputで、参照信号に対するノイズ元帥率を計算するための、参照信号のエネルギー
# NOISE_testのなかからランダムにtest_dataset_size個のデータを取り出す。    
for i in range(test_dataset_size) : 
    r = test_list[i]
    for j in range(4):
        test[j, i] = NOISE_test_ref[j,r:r+SEQUENCE_SIZE,0] ############################################################## 入力はref
        target_test[j, i] = NOISE_test_pri[j,r+SEQUENCE_SIZE+predict_sample_num-1,0]######################################################### 教師はpri
        # test_for_Einput_to_ref[j,i] = NOISE_test_ref[j,r+SEQUENCE_SIZE+predict_sample_num-1,0]

train_ref = torch.DoubleTensor(train_ref[:,:,np.newaxis])
print(f'train_ref {np.shape(train_ref)}')
target_train = torch.DoubleTensor(target_train)
test = torch.DoubleTensor(test[:,:,:,np.newaxis])
target_test = torch.DoubleTensor(target_test)

train_dataset = torch.utils.data.TensorDataset(train_ref, target_train)
test_dataset = [torch.utils.data.TensorDataset(test[0], target_test[0]),torch.utils.data.TensorDataset(test[1], target_test[1]),torch.utils.data.TensorDataset(test[2], target_test[2]),torch.utils.data.TensorDataset(test[3], target_test[3])]

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize , num_workers = 2)
testloader = [torch.utils.data.DataLoader(test_dataset[0], batch_size = 1, num_workers = 2),torch.utils.data.DataLoader(test_dataset[1], batch_size = 1, num_workers = 2),torch.utils.data.DataLoader(test_dataset[2], batch_size = 1, num_workers = 2),torch.utils.data.DataLoader(test_dataset[3], batch_size = 1, num_workers = 2)]

#----------------------------------------------------------------------------------------------------------


train_X, train_Y = dc.split_sequence(dataset, SEQUENCE_SIZE)
test_X, test_Y = dc.split_sequence(dataset_test, SEQUENCE_SIZE)
# for i in range(len(X)):
# 	print(X[i], Y[i])

trainX = torch.tensor(train_X[:, :, None], dtype=torch.float32)
trainY = torch.tensor(train_Y[:, None], dtype=torch.float32)
print(f'trainX {trainX.size()}')

testX = torch.tensor(test_X[:, :, None], dtype=torch.float32)
testY = torch.tensor(test_Y[:, None], dtype=torch.float32)
print(f'testX {testX.size()}')

time_steps = np.arange(SEQUENCE_SIZE, len(dataset_test))

# train_X, train_Y = train_ref, target_train
# test_X, test_Y = test[0,:,:,:], target_test[0,:,:]

# trainX = torch.tensor(train_X[:, :], dtype=torch.float32)
# trainY = torch.tensor(train_Y[:], dtype=torch.float32)
# print(f'trainX {trainX.size()}')

# testX = torch.tensor(test_X[:, :], dtype=torch.float32)
# testY = torch.tensor(test_Y[:], dtype=torch.float32)
# print(f'testX {testX.size()}')

# time_steps = np.arange(0, testX.size(0))


plt.ion()

figure, ax = plt.subplots(figsize=(8, 6))
plt.plot(time_steps, testY, label='Original Data')
(line1,) = ax.plot(time_steps, testY)


#-------------------------------HYPERPARAMETERS-------------------------------------------
lr=0.001
input_size=1
hidden_size= 64
num_layers=2
output_size=1
num_epochs = 500
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
        predicted, _, _ = model(testX)

        plt.title(f'Epoch [{epoch+1}/{num_epochs}]')
        line1.set_xdata(time_steps)
        line1.set_ydata(predicted.detach().numpy())
        figure.canvas.draw()
        figure.canvas.flush_events()
        

    if (epoch+1) % int(num_epochs/4) == 0:
        model.eval()
        with torch.no_grad():
            predicted, _, _ = model(testX)
            prediction.append(predicted.detach().numpy())


torch.save(model.state_dict(), 'test')

plt.ioff()

model.eval()
predicted, _, _ = model(testX)

with open('prediction.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(predicted.detach().numpy())

original = dataset_test[SEQUENCE_SIZE:]

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
