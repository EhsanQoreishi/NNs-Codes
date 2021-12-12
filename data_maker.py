import numpy as np
import pandas as pd
import glob as gb

schemes = [
    '1-Combination Output', '2-dx output', '3-Frequency indicator data',
          '4-Integration', '5-LBP', '6-SSF output', '7-Summation of Columns'
          , '9-Voltage Interval output', 'Origin'
        ]

ali

folder = 'Data-schemes/'
for name in schemes[-1:]:
    fld = folder+name+'/'
    excels_path = gb.glob(fld+'*.xlsx')
    X = []
    for path in excels_path:
        data = pd.read_excel(path)
        X.extend(data.to_numpy())
    X = np.array(X)
    np.save(fld+'X', X)


ali

# functions that prepare data for networks
def conv_data(x, lag=100,next = 1, trn_tst=(0.6, 0.8)):

    # Normalizing data
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x-mn)/(mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x)-lag - (next-1)):
        X.append(x[i:i+lag, 0])
        Y.append(x[i+lag+(next-1), 0])
    X = np.reshape(X, newshape=(len(X), 10, 10, 1))
    Y = np.array(Y)

    # shuffling data
    rand_list = np.random.permutation(len(X))
    X_per = np.zeros_like(X)
    Y_per = np.zeros_like(Y)
    for i in range(len(rand_list)):
        idx_per = rand_list[i]
        X_per[i] = X[idx_per]
        Y_per[i] = Y[idx_per]

    # Splitting data to train and test
    X_train = X_per[:int(trn_tst[0]*len(X_per))]
    Y_train = Y_per[:int(trn_tst[0]*len(Y_per))]
    X_val = X_per[int(trn_tst[0]*len(X_per)):int(trn_tst[1]*len(X_per))]
    Y_val = Y_per[int(trn_tst[0]*len(Y_per)):int(trn_tst[1]*len(Y_per))]
    X_test = X_per[int(trn_tst[1] * len(X_per)):]
    Y_test = Y_per[int(trn_tst[1] * len(Y_per)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def conv_data2(x, lag=100,next = 1, trn_tst=(0.6, 0.8)): # suit for only SensAnalys2

    """
    This function is suit for only SensAnalys2
    """
    # Normalizing data
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x-mn)/(mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x)-lag - (next-1)):
        X.append(x[i:i+lag, 0])
        Y.append(x[i+lag+(next-1), 0])
    X = np.reshape(X, newshape=(len(X), 10, 10, 1))
    Y = np.array(Y)

    # Splitting data to train and test
    X_train = X[:int(trn_tst[0]*len(X))]
    Y_train = Y[:int(trn_tst[0]*len(Y))]
    X_val =   X[int(trn_tst[0]*len(X)):int(trn_tst[1]*len(X))]
    Y_val =   Y[int(trn_tst[0]*len(Y)):int(trn_tst[1]*len(Y))]
    X_test =  X[int(trn_tst[1] * len(X)):]
    Y_test =  Y[int(trn_tst[1] * len(Y)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def mlp_data(x, lag=10, next=1, trn_tst=(0.6, 0.8)):
    # Normalizing data
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x - mn) / (mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x) - lag - (next-1)):
        X.append(x[i:i + lag, 0])
        Y.append(x[i + lag+(next-1), 0])
    X = np.reshape(X, newshape=(len(X), lag))
    Y = np.array(Y)

    # shuffling data
    rand_list = np.random.permutation(len(X))
    X_per = np.zeros_like(X)
    Y_per = np.zeros_like(Y)
    for i in range(len(rand_list)):
        idx_per = rand_list[i]
        X_per[i] = X[idx_per]
        Y_per[i] = Y[idx_per]

    # Splitting data to train and test and val
    X_train = X_per[:int(trn_tst[0] * len(X_per))]
    Y_train = Y_per[:int(trn_tst[0] * len(Y_per))]
    X_val = X_per[int(trn_tst[0] * len(X_per)):int(trn_tst[1] * len(X_per))]
    Y_val = Y_per[int(trn_tst[0] * len(Y_per)):int(trn_tst[1] * len(Y_per))]
    X_test = X_per[int(trn_tst[1] * len(X_per)):]
    Y_test = Y_per[int(trn_tst[1] * len(Y_per)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def mlp_data2(x, lag=10, next=1, trn_tst=(0.6, 0.8)):
    """
    This function is suit for only SensAnalys2
    :param x:
    :param lag:
    :param next:
    :param trn_tst:
    :return:
    """
    # Normalizing data
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x - mn) / (mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x) - lag - (next-1)):
        X.append(x[i:i + lag, 0])
        Y.append(x[i + lag+(next-1), 0])
    X = np.reshape(X, newshape=(len(X), lag))
    Y = np.array(Y)

    # Splitting data to train and test and val
    X_train = X[:int(trn_tst[0] * len(X))]
    Y_train = Y[:int(trn_tst[0] * len(Y))]
    X_val =   X[int(trn_tst[0] * len(X)):int(trn_tst[1] * len(X))]
    Y_val =   Y[int(trn_tst[0] * len(Y)):int(trn_tst[1] * len(Y))]
    X_test =  X[int(trn_tst[1] * len(X)):]
    Y_test =  Y[int(trn_tst[1] * len(Y)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def lstm_data(x, lag=10, next=1, trn_tst=(0.6, 0.8)):
    # Normalizing data
    n_feature = 1
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x - mn) / (mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x) - lag - (next-1)):
        X.append(x[i:i + lag, 0])
        Y.append(x[i + lag + (next-1), 0])
    X = np.reshape(X, newshape=(len(X), lag, n_feature))
    Y = np.array(Y)

    # shuffling data
    rand_list = np.random.permutation(len(X))
    X_per = np.zeros_like(X)
    Y_per = np.zeros_like(Y)
    for i in range(len(rand_list)):
        idx_per = rand_list[i]
        X_per[i] = X[idx_per]
        Y_per[i] = Y[idx_per]

    # Splitting data to train and test and val
    X_train = X_per[:int(trn_tst[0] * len(X_per))]
    Y_train = Y_per[:int(trn_tst[0] * len(Y_per))]
    X_val = X_per[int(trn_tst[0] * len(X_per)):int(trn_tst[1] * len(X_per))]
    Y_val = Y_per[int(trn_tst[0] * len(Y_per)):int(trn_tst[1] * len(Y_per))]
    X_test = X_per[int(trn_tst[1] * len(X_per)):]
    Y_test = Y_per[int(trn_tst[1] * len(Y_per)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

def lstm_data2(x, lag=10, next=1, trn_tst=(0.6, 0.8)):
    """
    This function is suit only for SensAnalys2
    :param x:
    :param lag:
    :param next:
    :param trn_tst:
    :return:
    """
    # Normalizing data
    n_feature = 1
    mn = np.min(x, axis=0)
    mx = np.max(x, axis=0)
    x = (x - mn) / (mx - mn)

    # Forming data somehow being suite for conv network
    X = []
    Y = []
    for i in range(len(x) - lag - (next-1)):
        X.append(x[i:i + lag, 0])
        Y.append(x[i + lag + (next-1), 0])
    X = np.reshape(X, newshape=(len(X), lag, n_feature))
    Y = np.array(Y)

    # Splitting data to train and test and val
    X_train = X[:int(trn_tst[0] * len(X))]
    Y_train = Y[:int(trn_tst[0] * len(Y))]
    X_val =   X[int(trn_tst[0] * len(X)):int(trn_tst[1] * len(X))]
    Y_val =   Y[int(trn_tst[0] * len(Y)):int(trn_tst[1] * len(Y))]
    X_test =  X[int(trn_tst[1] * len(X)):]
    Y_test =  Y[int(trn_tst[1] * len(Y)):]

    return (X_train, Y_train, X_val, Y_val, X_test, Y_test)

# Preparing data

scheme_no = 1
folder = 'Data-schemes/'
net1 = 'Conv/'
net2 = 'DFCN/'
net3 = 'LSTM/'
# net4 = 'RBF/'

all_data_conv = {}
all_data_mlp = {}
next_steps = [1, 3, 5, 7, 9, 11, 13, 15]
for name in schemes[-1:]:
    print(name)
    fld = folder+name+'/'
    try:
        X = np.load(fld+'X.npy')[:500000]
    except:
        X = np.load(fld+'X.npy')
    for n in next_steps:
        fld2 = '/' + str(n) + ' next step/'
        c_data = conv_data2(X, next=n)
        np.save('Features/' + net1 + name + fld2 + '/X_train2', c_data[0])
        np.save('Features/' + net1 + name + fld2 + '/Y_train2', c_data[1])
        np.save('Features/' + net1 + name + fld2 + '/X_val2', c_data[2])
        np.save('Features/' + net1 + name + fld2 + '/Y_val2', c_data[3])
        np.save('Features/' + net1 + name + fld2 + '/X_test2', c_data[4])
        np.save('Features/' + net1 + name + fld2 + '/Y_test2', c_data[5])

        m_data = mlp_data2(X, next=n)
        np.save('Features/' + net2 + name+ fld2 + '/X_train2', m_data[0])
        np.save('Features/' + net2 + name+ fld2 + '/Y_train2', m_data[1])
        np.save('Features/' + net2 + name+ fld2 + '/X_val2', m_data[2])
        np.save('Features/' + net2 + name+ fld2 + '/Y_val2', m_data[3])
        np.save('Features/' + net2 + name+ fld2 + '/X_test2', m_data[4])
        np.save('Features/' + net2 + name+ fld2 + '/Y_test2', m_data[5])

        ls_data = lstm_data2(X, next=n)
        np.save('Features/' + net3 + name + fld2 + '/X_train2', ls_data[0])
        np.save('Features/' + net3 + name + fld2 + '/Y_train2', ls_data[1])
        np.save('Features/' + net3 + name + fld2 + '/X_val2', ls_data[2])
        np.save('Features/' + net3 + name + fld2 + '/Y_val2', ls_data[3])
        np.save('Features/' + net3 + name + fld2 + '/X_test2', ls_data[4])
        np.save('Features/' + net3 + name + fld2 + '/Y_test2', ls_data[5])

