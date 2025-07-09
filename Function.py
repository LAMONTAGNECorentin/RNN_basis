import Myfunction
import torch
import numpy as np



def pre_treatement_set1(output_size = 1, test_dataset_size = 10000, TRAIN_SET_NUM = 10, SEQUENCE_SIZE = 20, Ts_train = 0.001, predict_sample_num = 1, batchsize = 1, fs_resample = 44.1):

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

    return train_ref, target_train, test, target_test

def pre_treatement_set2(output_size = 1, test_dataset_size = 10000, TRAIN_SET_NUM = 10, SEQUENCE_SIZE = 20, Ts_train = 0.001, predict_sample_num = 1, batchsize = 1, fs_resample = 44.1):
    import scipy

    fs_resample_khz = fs_resample
    fs_resample = fs_resample*10**3

    Ts = float(1/(fs_resample))             # frequency to periode
    train_dataset_size = int(Ts_train/Ts) 

    factory2 = scipy.io.loadmat('../dataset/SPIB/orig_19.98kHz_mat/factory2.mat')
    factory2_ref = factory2['factory2']

    factory2_ref = factory2_ref/(2**15)

    kind = 'cubic'
    fs_orig=19.98*10**3             # 19.98Khz
    size = int(len(factory2) * fs_resample / fs_orig)

    # 一時保存用配列
    factory2_ref_temp, factory2_ref_temp, volvo_ref_temp, destroyerengine_ref_temp = np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1]),np.zeros([size,1])
    # アップサンプリング実行を一時保存用配列へ
    factory2_ref_temp = Myfunction.UPSAMPLING(signal=factory2_ref,fs_orig=fs_orig, fs_resample=fs_resample, KIND=kind)
    # この後使う配列へ代入
    factory2_ref = np.zeros([size,1])

    # factory2, factory2, volvo, destroyerengine =factory2_temp, factory2_temp, volvo_temp, destroyerengine_temp
    factory2_ref = factory2_ref_temp.copy()
    # 一次信号の変数に、参照信号の値を代入する(_pri = _ref)
    factory2_pri = factory2_ref_temp.copy()

    divide = int(factory2_pri.shape[0]*0.5)

    NOISE_train_ref = np.vstack([factory2_ref[:divide], factory2_ref[:divide], factory2_ref[:divide], factory2_ref[:divide]])
    NOISE_train_pri = np.vstack([factory2_pri[:divide], factory2_pri[:divide], factory2_pri[:divide], factory2_pri[:divide]])

    # ################################################################################################
    # (4, 5203786, 1)
    NOISE_test_ref = np.stack([factory2_ref[divide:], factory2_ref[divide:], factory2_ref[divide:], factory2_ref[divide:]], axis=0)
    NOISE_test_pri = np.stack([factory2_pri[divide:], factory2_pri[divide:], factory2_pri[divide:], factory2_pri[divide:]], axis=0)

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

    return train_ref, target_train, test, target_test
