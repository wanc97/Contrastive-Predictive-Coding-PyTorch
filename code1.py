# 导入必要库
from sklearn.svm import OneClassSVM
from sklearn.svm import NuSVC
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import copy

# 定义函数获得无故障及21种故障的训练集与测试集
def get_alldata():
    def set_to_one(lis):
        lis = np.array(lis)
        setone = [[], []]
        t = lis.T
        for i in range(len(t)):
            setone[0].append(np.mean(t[i]))
            setone[1].append(np.std(t[i]))
        for i in range(len(lis)):
            for j in range(len(lis[0])):
                lis[i][j] = (lis[i][j] - setone[0][j]) / setone[1][j]
        return lis, setone

    def to_one(lis, setone):
        lis = np.array(lis)
        for i in range(len(lis)):
            for j in range(len(lis[0])):
                lis[i][j] = (lis[i][j] - setone[0][j]) / setone[1][j]
        return lis

    def get_data(num):
        def dat_to_list_train0(path):
            li = []
            with open(path, mode='r') as f:
                b = f.readline()
                c = b.split()
                for i in range(len(c)):
                    a = []
                    a.append(float(c[i]))
                    li.append(a)
                b = f.readline()
                while b != None and b != '':
                    c = b.split()
                    for i in range(len(c)):
                        li[i].append(float(c[i]))
                    b = f.readline()
            return li

        def dat_to_list(path):
            li = []
            with open(path, mode='r') as f:
                b = f.readline()
                while b != None and b != '':
                    c = b.split()
                    for i in range(len(c)):
                        c[i] = float(c[i])
                    li.append(c)
                    b = f.readline()
            return li

        if num < 0 or num > 21 or type(num) != int:
            print('the number of data is error')
        else:
            if num < 10:
                num = '0' + str(num)
            else:
                num = str(num)
            if num == '00':
                train = dat_to_list_train0('data/d' + num + '.dat')
            else:
                train = dat_to_list('data/d' + num + '.dat')
            test = dat_to_list('data/d' + num + '_te.dat')
            return train, test

    train = []
    test = []
    tr, te = get_data(0)
    tr, setone = set_to_one(tr)
    train.append(tr.tolist())
    test.append(to_one(te, setone).tolist())
    for i in range(1,22):
        tr,te = get_data(i)
        te = to_one(te, setone)
        train.append(to_one(tr, setone).tolist())
        test.append([te.tolist()[0:160],te.tolist()[160:]])
    return train,test

# 定义函数用KNN去除样本中的可疑异常点，用于OCSVM，本代码暂时未使用此函数
def get_newdata(lis, k):
    def distance(lis):
        m, n = np.shape(lis)
        distan = np.zeros((m, m))
        lis = np.mat(lis)
        for i in range(m):
            for j in range(i + 1, m):
                d = lis[i] - lis[j]
                distan[i][j] = np.sqrt(d * d.T)
                distan[j][i] = distan[i][j]
        return distan

    def avr_k(lis, k):
        k = k + 1
        m, _ = np.shape(lis)
        avr = []
        for i in range(m):
            a = heapq.nsmallest(k, lis[i])
            a.pop(0)
            avr.append(np.mean(a))
        return avr

    arg = np.argsort(avr_k(distance(lis), k))
    newdata = []
    for i in range(int(len(arg)*0.9)):
        newdata.append(lis[arg[i]])
    return newdata

# 定义函数获得训练集与测试集的标签
def get_label():
    ytrain=[]
    ytest=[]
    for i in range(500):
        ytrain.append(0)
    for i in range(480):
        ytrain.append(1)
    for i in range(160):
        ytest.append(0)
    for i in range(800):
        ytest.append(1)
    return ytrain, ytest

# 定义函数对多个数据集进行PCA降维并可视化展示
def show_by_PCA(lis):
    a = len(lis)
    if a <=17:
        x = []
        for i in lis:
            x = x + i
        pca = PCA(n_components=3)     #加载PCA算法，设置降维后主成分数目为2
        reduced_x = pca.fit_transform(x)  # 对样本进行降维
        lenall = 0
        colors = ['k','r','y','g','c','b','m','grey','maroon','tan','gold','lime','lightbiue','navy','blueviolet','violet','pink']
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(a):
            length = len(lis[i]) + lenall
            ax.scatter(reduced_x[lenall:length,0], reduced_x[lenall:length,1],reduced_x[lenall:length,2], c=colors[i], marker='x')
            lenall = length
        plt.show()
    else:
        print('Too many categories to display')

# 定义函数对给定训练集进行SVM建模并根据测试集返回准确率
def fd_svm(train, test, ytrain, ytest):
    clf = NuSVC()
    clf.fit(train, ytrain)
    return clf.score(test, ytest)

# 定义函数对给定训练集进行SVM-sequence建模并根据测试集返回准确率
def fd_svm_time(train,test,ytrain, ytest,seq):
    for i in range(len(train) - seq + 1):
        for j in range(1, seq):
            train[i] = train[i] + train[i+j]
    train = train[:- seq + 1]
    train = np.array(train).astype('float64')
    train_y = np.array(ytrain[seq - 1:]).astype('float64')
    for i in range(len(test) - seq + 1):
        for j in range(1, seq):
            test[i] = test[i] + test[i+j]
    test = test[:- seq + 1]
    test = np.array(test).astype('float64')
    test_y = np.array(ytest[seq - 1:]).astype('float64')
    clf = NuSVC()
    clf.fit(train, train_y)
    return clf.score(test, test_y)

# 定义函数对给定训练集进行SVM-sequence-prior建模并根据测试集返回准确率
def fd_svm_time_prior(train,test,ytrain, ytest,seq,k):
    for i in range(len(train) - seq + 1):
        for j in range(1, seq):
            train[i] = train[i] + train[i+j]
    train = train[:- seq + 1]
    train = np.array(train).astype('float64')
    train_y = np.array(ytrain[seq - 1:]).astype('float64')
    for i in range(len(test) - seq + 1):
        for j in range(1, seq):
            test[i] = test[i] + test[i+j]
    test = test[:- seq + 1]
    test = np.array(test).astype('float64')
    test_y = np.array(ytest[seq - 1:]).astype('float64')
    clf = NuSVC()
    clf.fit(train, train_y)
    predict_y = clf.predict(test)
    # return clf.predict(test)
    predict_y = list(predict_y)
    for i in range(len(predict_y) - k + 1):
        if 0 in set(predict_y[i:i + k]):
            continue
        else:
            for j in range(i + k, len(predict_y)):
                predict_y[j] = 1
            break

    for i in range(len(predict_y)):
        if predict_y[i] == test_y[i]:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
    return np.average(predict_y)

# 定义函数对给定训练集进行LSTM建模并根据测试集返回准确率
def fd_lstm(train,test,ytrain, ytest,seq):
    train_x = []
    for i in range(len(train) - seq +1):
        train_x.append(train[i: i + seq])
    train_x = np.array(train_x).astype('float64')
    train_y = np.array(ytrain[seq-1:]).astype('float64')
    test_x = []
    for i in range(len(test) - seq + 1):
        test_x.append(test[i: i + seq])
    test_x = np.array(test_x).astype('float64')
    test_y = np.array(ytest[seq - 1:]).astype('float64')

    model = Sequential()
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1,activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.1)
    predict_y = model.predict(test_x)
    predict_y[predict_y < 0.5] = 0
    predict_y[predict_y > 0.5] = 1

    for i in range(len(predict_y)):
        if predict_y[i] == test_y[i]:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
    return np.average(predict_y)

# 定义函数对给定训练集进行LSTM-prior建模并根据测试集返回准确率
def fd_lstm_prior(train, test, ytrain, ytest,seq, k):
    train_x = []
    for i in range(len(train) - seq + 1):
        train_x.append(train[i: i + seq])
    train_x = np.array(train_x).astype('float64')
    train_y = np.array(ytrain[seq - 1:]).astype('float64')
    test_x = []
    for i in range(len(test) - seq + 1):
        test_x.append(test[i: i + seq])
    test_x = np.array(test_x).astype('float64')
    test_y = np.array(ytest[seq - 1:]).astype('float64')

    model = Sequential()
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.1)
    predict_y = model.predict(test_x)
    predict_y[predict_y < 0.5] = 0
    predict_y[predict_y > 0.5] = 1

    predict_y = list(predict_y.reshape((predict_y.size,)))
    for i in range(len(predict_y)-k+1):
        if 0 in set(predict_y[i:i+k]):
            continue
        else:
            for j in range(i+k,len(predict_y)):
                predict_y[j] = 1
            break

    for i in range(len(predict_y)):
        if predict_y[i] == test_y[i]:
            predict_y[i] = 1
        else:
            predict_y[i] = 0
    return np.average(predict_y)

# 定义函数对给定训练集进行LSTM建模并根据测试集返回所有样本的预测值，用于快速获得批量模型的准确率
def fd_lstm_test(train,test,ytrain, ytest,seq):
    train_x = []
    for i in range(len(train) - seq +1):
        train_x.append(train[i: i + seq])
    train_x = np.array(train_x).astype('float64')
    train_y = np.array(ytrain[seq-1:]).astype('float64')
    test_x = []
    for i in range(len(test) - seq + 1):
        test_x.append(test[i: i + seq])
    test_x = np.array(test_x).astype('float64')
    test_y = np.array(ytest[seq - 1:]).astype('float64')

    model = Sequential()
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(input_dim=52, output_dim=50, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=1,activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y, batch_size=100, nb_epoch=50, validation_split=0.1)
    predict_y = model.predict(test_x)
    predict_y[predict_y < 0.5] = 0
    predict_y[predict_y > 0.5] = 1
    return predict_y,test_y

# 获得数据集和标签
train, test = get_alldata()
ytrain, ytest = get_label()

# 对五种算法进行进行实验，获取并保存各种情况下的准确率
result_svm = []
for i in range(1, 22):
    result_svm.append({'Algorithm':'SVM','Fault source':i,'Prediction accuracy':fd_svm(train[0] + train[i], test[i][0] + test[i][1], ytrain, ytest)})
df_result_svm = pd.DataFrame(result_svm)
df_result_svm.to_csv(r'result/'  + 'result_svm' + '.csv')

result_svm_time = []
for i in range(1, 22):
    for j in range(5, 50, 5):
        result_svm_time.append({'Algorithm':'SVM_time','Fault source':i,'sequence':j,'Prediction accuracy':fd_svm_time(train[0] + train[i], test[i][0] + test[i][1], ytrain, ytest,j)})
df_result_svm_time = pd.DataFrame(result_svm_time)
df_result_svm_time.to_csv(r'result/'  + 'result_svm_time' + '.csv')

result_svm_time_prior = []
for i in range(1, 22):
    for j in range(5, 50, 5):
        for k in range(10, 30, 5):
            result_svm_time_prior.append({'Algorithm': 'SVM_time_prior', 'Fault source': i, 'sequence': j,'Prior number':k,
                                    'Prediction accuracy': fd_svm_time_prior(train[0] + train[i], test[i][0] + test[i][1],
                                                                       ytrain, ytest, j, k)})
df_result_svm_time_prior = pd.DataFrame(result_svm_time_prior)
df_result_svm_time_prior.to_csv(r'result/'  + 'result_svm_time_prior' + '.csv')

result_lstm = []
for i in range(1, 22):
    for j in range(5, 50, 5):
        result_lstm.append({'Algorithm':'LSTM','Fault source':i,'sequence':j,'Prediction accuracy':fd_lstm(train[0] + train[i], test[i][0] + test[i][1], ytrain, ytest,j)})
df_result_lstm = pd.DataFrame(result_lstm)
df_result_lstm.to_csv(r'result/'  + 'result_lstm' + '.csv')

result_lstm_prior = []
for i in range(1, 22):
    for j in range(5, 50, 5):
        for k in range(10, 30, 5):
            result_lstm_prior.append({'Algorithm': 'LSTM_prior', 'Fault source': i, 'sequence': j,'Prior number':k,
                                    'Prediction accuracy': fd_lstm_prior(train[0] + train[i], test[i][0] + test[i][1],
                                                                       ytrain, ytest, j, k)})
df_result_lstm_prior = pd.DataFrame(result_lstm_prior)
df_result_lstm_prior.to_csv(r'result/'  + 'result_lstm_prior' + '.csv')

# 用于快速获得并保存LSTM和LSTM-prior两类模型的准确率，可选
# result_lstm = []
# result_lstm_prior = []
# for i in range(1, 22):
#     for j in range(5, 50, 5):
#         predict_y,test_y = fd_lstm_test(train[0] + train[i], test[i][0] + test[i][1], ytrain, ytest,j)
#         aaa=copy.deepcopy(predict_y)
#         for z in range(len(aaa)):
#             if aaa[z] == test_y[z]:
#                 aaa[z] = 1
#             else:
#                 aaa[z] = 0
#         result_lstm.append({'Algorithm':'LSTM','Fault source':i,'sequence':j,'Prediction accuracy':np.average(aaa)})
#
#         predict_y = list(predict_y.reshape((predict_y.size,)))
#         for k in range(10, 30, 5):
#             aaa = copy.deepcopy(predict_y)
#             for z in range(len(aaa) - k + 1):
#                 if 0 in set(aaa[z:z + k]):
#                     continue
#                 else:
#                     for g in range(z + k, len(aaa)):
#                         aaa[g] = 1
#                     break
#
#             for z in range(len(aaa)):
#                 if aaa[z] == test_y[z]:
#                     aaa[z] = 1
#                 else:
#                     aaa[z] = 0
#             result_lstm_prior.append({'Algorithm': 'LSTM_prior', 'Fault source': i, 'sequence': j, 'Prior number': k,
#                                       'Prediction accuracy': np.average(aaa)})
# df_result_lstm = pd.DataFrame(result_lstm)
# df_result_lstm.to_csv(r'result/'  + 'result_lstm' + '.csv')
# df_result_lstm_prior = pd.DataFrame(result_lstm_prior)
# df_result_lstm_prior.to_csv(r'result/'  + 'result_lstm_prior' + '.csv')

result = []
result = result + result_svm + result_svm_time + result_svm_time_prior + result_lstm + result_lstm_prior
df_result = pd.DataFrame(result)
df_result.to_csv(r'result/'  + 'result' + '.csv')

# 用PCA可视化各类故障的状态分布
for i in range(1, 22):
    show_by_PCA([train[0], train[i]])

# 对各个模型的数据进行处理得到模型最佳表现并保存
df_result_svm_time_gy = df_result_svm_time[['Fault source','Prediction accuracy']].groupby(by='Fault source',as_index=False).max()
df_result_svm_time_gy.to_csv(r'result/'  + 'df_result_svm_time_gy' + '.csv')
df_result_svm_time_prior_gy = df_result_svm_time_prior[['Fault source','Prediction accuracy']].groupby(by='Fault source',as_index=False).max()
df_result_svm_time_prior_gy.to_csv(r'result/'  + 'df_result_svm_time_prior_gy' + '.csv')
df_result_lstm_gy = df_result_lstm[['Fault source','Prediction accuracy']].groupby(by='Fault source',as_index=False).max()
df_result_lstm_gy.to_csv(r'result/'  + 'df_result_lstm_gy' + '.csv')
df_result_lstm_prior_gy = df_result_lstm_prior[['Fault source','Prediction accuracy']].groupby(by='Fault source',as_index=False).max()
df_result_lstm_prior_gy.to_csv(r'result/'  + 'df_result_lstm_prior_gy' + '.csv')