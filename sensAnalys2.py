import numpy as np
from keras import models
import joblib
import matplotlib.pyplot as plt
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


nexts = [1, 3, 5, 7, 9, 11, 13, 15]
schemes = [
            '1-Combination Output', '2-dx output', '3-Frequency indicator data',
            '4-Integration', '5-LBP', '6-SSF output', '7-Summation of Columns'
           ,'9-Voltage Interval output', 'Origin'
        ]

schemes2 = [
            'Combination of columns', 'Derivative', 'Frequency',
            'Integral', 'LBP', 'Simple sampling with assigned frequency', 'Summation of Columns'
           ,'Voltage Interval', 'Simple sampling'
]

nets = ['Conv/', 'DFCN/', 'RBF/', 'SVM/', 'LSTM/']

for sn in range(len(schemes)):
    scheme = schemes[sn]
    predicts = []
    Y_tests = []
    for net in nets:

        print('Starting : ', scheme, '---', net[:-1])

        # Loading Data
        Y_test = np.load('Features/' + 'DFCN/' + scheme + '/1 next step/Y_test2.npy')[:120]
        Y_test = np.reshape(Y_test, newshape=(Y_test.shape[0], 1))
        if net =='Conv/':
            X_test = []
            temp = []
            lag = 100
            for i in range(len(Y_test) - lag):
                X_test.append(Y_test[i:i + lag, 0])
                temp.append(Y_test[i + lag, 0])
            X_test = np.reshape(X_test, newshape=(len(X_test), 10, 10, 1))
            Y_test = np.array(temp)
            Y_tests.append(Y_test)

        elif net == 'LSTM/':
            X_test = []
            temp = []
            lag = 10
            for i in range(90, len(Y_test) - lag):
                X_test.append(Y_test[i:i + lag, 0])
                temp.append(Y_test[i + lag, 0])
            X_test = np.reshape(X_test, newshape=(len(X_test), lag, 1))
            Y_test = np.array(temp)
            Y_tests.append(Y_test)

        else:
            X_test = []
            temp = []
            lag = 10
            for i in range(90, len(Y_test) - lag):
                X_test.append(Y_test[i:i + lag, 0])
                temp.append(Y_test[i + lag, 0])
            X_test = np.reshape(X_test, newshape=(len(X_test), lag))
            Y_test = np.array(temp)
            Y_tests.append(Y_test)

        # Loading model
        if net not in ['SVM/', 'RBF/']:
            M1 = models.load_model('Results/' + net + scheme + '/1 next step/trained2.h5')
        else:
            M1 = joblib.load('Results/' + net + scheme + '/1 next step/trained2.joblib')

        # First mode --> this is common mode (:
        abs_error1 = []
        for i in range(X_test.shape[0]):
            x = X_test[i]
            y = Y_test[i]
            if len(x.shape)==3:
                x = np.reshape(x, newshape=(1, x.shape[0] , x.shape[1], x.shape[2]))
            elif len(x.shape)==2:
                x = np.reshape(x, newshape=(1, x.shape[0] , x.shape[1]))
            else:
                x = np.reshape(x, newshape=(1, x.shape[0]))

            pred = M1.predict(x)[0]
            abs_error1.append(np.abs((y-pred)))

        # Second Mode --> this is an unusual mode ):
        abs_error2 = []
        preds = []
        for i in range(X_test.shape[0]):
            x = X_test[i]
            y = Y_test[i]
            if len(x.shape)==3:   # conv data
                pred = M1.predict(np.reshape(x, newshape=(1, 10, 10, 1)))[0]
                if i < X_test.shape[0]-100:
                    for j in range(100):
                        temp = 100 - (j+1)
                        row = temp//10
                        col = temp%10
                        X_test[i+j+1, row, col, 0] = pred
                preds.append(pred)
                abs_error2.append(np.abs((y-pred)))

            elif len(x.shape)==2:  # lstm data
                pred = M1.predict(np.reshape(x, newshape=(1, 10, 1)))[0]
                if i < X_test.shape[0]-x.shape[0]:
                    for j in range(x.shape[0]):
                        X_test[i+j+1, x.shape[0]-(j+1), 0] = pred
                preds.append(pred)
                abs_error2.append(np.abs((y-pred)))

            elif len(x.shape)==1:  # mlp data
                pred = M1.predict(np.reshape(x, newshape=(1, 10)))[0]
                if i < X_test.shape[0]-x.shape[0]:
                    for j in range(x.shape[0]):
                        X_test[i+j+1, x.shape[0]-(j+1)] = pred
                preds.append(pred)
                abs_error2.append(np.abs((y-pred)))

        predicts.append(preds)

        # plt.plot(abs_error1, color='blue', label='abs1')
        # plt.plot(abs_error2, color='red', label='abs2')
        # plt.xlabel('sample number')
        # plt.ylabel('absolute error')
        # plt.title(net[:-1])
        # plt.legend()
        # plt.show()
    temp = np.arange(0, len(Y_test))
    temp2 = []
    for elm in temp:
        temp2.append(str(elm))
    plt.plot(temp2, Y_test, label='actual', marker='*')
    for k in range(len(nets)):
        net = nets[k][:-1]
        plt.plot(temp2, predicts[k], label=net, marker='*')
    plt.xlabel('Time steps')
    plt.ylabel('Normalized signal value')
    plt.title(schemes2[sn])
    plt.legend()
    plt.savefig('Results/Sens2/'+schemes2[sn]+'.jpg', dpi=540)
    plt.show()

