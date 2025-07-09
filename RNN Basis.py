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
import time
import Function as f
import pandas as pd

torch.manual_seed(100)

# dataset = dc.sinus(start=0, end=100, step=0.1, amplitude=5, style=0)
# dataset_test = dc.sinus(start=0.1, end=100.1, step=0.1, amplitude=5, style=0)

# dataset = dc.sinus_nonlinear(start=0, end=10, step=0.01, amplitude=5, style=3)
# dataset_test = dc.sinus_nonlinear(start=0.01, end=10.01, step=0.01, amplitude=5, style=3)

#-------------------------------HYPERPARAMETERS-------------------------------------------
lr=0.001
input_size=1
HIDDEN_SIZE= [16, 32, 64]
NUM_LAYERS= [1, 2, 3]
output_size=1
EPOCH = [1500]
h0, c0 = None, None
#----------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------
output_size = 1
TEST_DATASET_SIZE = [10000]
train_set_num = [10, 20, 50]
SEQUENCE_SIZE = [10, 20, 50]
Ts_train = 0.001                # Training period
predict_sample_num = 1
batchsize = 1
fs_resample = 44.1              # Frequency

Tot_iter = len(HIDDEN_SIZE)*len(NUM_LAYERS)*len(TEST_DATASET_SIZE)*len(SEQUENCE_SIZE)*len(train_set_num)

i=0
loss_LOG, test_dataset_size_LOG, sequence_size_LOG, lr_LOG, hidden_size_LOG, num_layers_LOG, num_epochs_LOG, Time_LOG = [],[],[],[],[],[],[],[]
for TRAIN_SET_NUM in train_set_num:
    for sequence_size in SEQUENCE_SIZE:
        for test_dataset_size in TEST_DATASET_SIZE:
            train_ref, target_train, test, target_test = f.pre_treatement(output_size, test_dataset_size, TRAIN_SET_NUM, sequence_size, Ts_train, predict_sample_num, batchsize, fs_resample)

            #----------------------------------------------------------------------------------------------------------


            # train_X, train_Y = dc.split_sequence(dataset, SEQUENCE_SIZE)
            # test_X, test_Y = dc.split_sequence(dataset_test, SEQUENCE_SIZE)
            # # for i in range(len(X)):
            # # 	print(X[i], Y[i])

            # trainX = torch.tensor(train_X[:, :, None], dtype=torch.float32)
            # trainY = torch.tensor(train_Y[:, None], dtype=torch.float32)
            # print(f'trainX {trainX.size()}')

            # testX = torch.tensor(test_X[:, :, None], dtype=torch.float32)
            # testY = torch.tensor(test_Y[:, None], dtype=torch.float32)
            # print(f'testX {testX.size()}')

            # time_steps = np.arange(SEQUENCE_SIZE, len(dataset_test))

            train_X, train_Y = train_ref, target_train
            test_X, test_Y = test[0,:,:,:], target_test[0,:,:]

            trainX = torch.tensor(train_X[:, :], dtype=torch.float32)
            trainY = torch.tensor(train_Y[:], dtype=torch.float32)
            testX = torch.tensor(test_X[:, :], dtype=torch.float32)
            testY = torch.tensor(test_Y[:], dtype=torch.float32)

            time_steps = np.arange(0, testX.size(0))


            # plt.ion()

            # figure, ax = plt.subplots(figsize=(8, 6))
            # plt.plot(time_steps, testY, label='Original Data')
            # (line1,) = ax.plot(time_steps, testY)

            for num_epochs in EPOCH:
                for num_layers in NUM_LAYERS:
                    for hidden_size in HIDDEN_SIZE:
                        
                        i+=1
                        print(f'- - - - - - - - Iteration {i}/{Tot_iter} - - - - - - - -')

                        Time = time.time()
                        h0, c0 = None, None

                        Mymodel = model.LSTM_model(input_size, hidden_size, num_layers, output_size)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(Mymodel.parameters(), lr)

                        Loss_history = []
                        prediction = []

                        for epoch in range(num_epochs):
                            Mymodel.train()
                            #add scheduler here
                            optimizer.zero_grad()

                            outputs, h0, c0 = Mymodel(trainX, h0, c0)

                            loss = criterion(outputs, trainY)
                            loss.backward()
                            optimizer.step()

                            h0 = h0.detach()
                            c0 = c0.detach()

                            Loss_history.append(loss.item())

                            if (epoch+1) % 100 == 0:
                                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                                Mymodel.eval()
                                predicted, _, _ = Mymodel(testX)

                                # plt.title(f'Epoch [{epoch+1}/{num_epochs}]')
                                # line1.set_xdata(time_steps)
                                # line1.set_ydata(predicted.detach().numpy())
                                # figure.canvas.draw()
                                # figure.canvas.flush_events()
                                

                            if (epoch+1) % int(num_epochs/4) == 0:
                                Mymodel.eval()
                                with torch.no_grad():
                                    predicted, _, _ = Mymodel(testX)
                                    prediction.append(predicted.detach().numpy())

                        Time = time.time()-Time
                        torch.save(Mymodel.state_dict(), f'Loss_{loss.item()}_TDS_{test_dataset_size}_TSN_{train_set_num}_SS_{sequence_size}_lr_{lr}_hs_{hidden_size}_num-layers_{num_layers}_epochs_{num_epochs}_time_{Time}')
                        loss_LOG.append(loss.item())
                        test_dataset_size_LOG.append(test_dataset_size)
                        sequence_size_LOG.append(sequence_size)
                        lr_LOG.append(lr)
                        hidden_size_LOG.append(hidden_size)
                        num_layers_LOG.append(num_layers)
                        num_epochs_LOG.append(num_epochs)
                        Time_LOG.append(Time)
                        # plt.ioff()

                        Mymodel.eval()
                        predicted, _, _ = Mymodel(testX)

                        with open('prediction.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(predicted.detach().numpy())

                        # original = dataset_test[sequence_size:]

                        # plt.figure(figsize=(12, 6))
                        # plt.subplot(1,2,1)
                        # plt.plot(Loss_history, label='Loss')
                        # plt.title('LSTM Loss evolution')
                        # plt.xlabel('Epoch')
                        # plt.ylabel('Value')
                        # plt.legend()

                        # plt.subplot(1,2,2)
                        # plt.plot(time_steps, testY, label='Original Data')
                        # plt.plot(time_steps, predicted.detach().numpy(), label='Predicted Data', linestyle='--')
                        # plt.title('LSTM Model Predictions vs. Original Data')
                        # plt.xlabel('Time Step')
                        # plt.ylabel('Value')
                        # plt.legend()
                        # plt.show()

data = {
    "Loss":loss_LOG,
    "TDS":test_dataset_size_LOG,
    "SS":sequence_size_LOG,
    "lr":lr_LOG,
    "hs":hidden_size_LOG,
    "num-layers":num_layers_LOG,
    "epochs":num_epochs_LOG,
    "time":Time_LOG
}

df = pd.DataFrame(data)
df.to_excel("batch.xlsx", index=False)

