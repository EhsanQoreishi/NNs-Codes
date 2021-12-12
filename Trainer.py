import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

from keras import models
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)
from keras import callbacks

nexts = [1, 3, 5, 7, 9, 11, 13, 15]
schemes = [
            '1-Combination Output', '2-dx output', '3-Frequency indicator data',
            '4-Integration', '5-LBP', '6-SSF output', '7-Summation of Columns'
           ,'9-Voltage Interval output', 'Origin'
        ]
nets = ['Conv/', 'DFCN/', 'RBF/', 'SVM/', 'LSTM/']

for scheme in schemes[-1:]:
    scheme_R2 = []  # saving R2 for each net by changing next steps --> shape = 2D
    for net in nets:
        net_R2 = []  # saving R2 for by changing next steps --> shape = 1D
        for next in nexts:

            next_fld = '/' + str(next) + ' next step/'
            print('Starting : ', scheme, '---', net[:-1], '---', next_fld[:-1])

            # Loading Data
            if net in ['Conv/', 'LSTM/']:
                X_train = np.load('Features/' + net + scheme + next_fld + '/X_train2.npy')[:10000]
                Y_train = np.load('Features/' + net + scheme + next_fld + '/Y_train2.npy')[:10000]
                X_val =   np.load('Features/' + net + scheme + next_fld + '/X_val2.npy')[:1000]
                Y_val =   np.load('Features/' + net + scheme + next_fld + '/Y_val2.npy')[:1000]
                X_test =  np.load('Features/' + net + scheme + next_fld + '/X_test2.npy')[:1000]
                Y_test =  np.load('Features/' + net + scheme + next_fld + '/Y_test2.npy')[:1000]
            else:
                X_train = np.load('Features/' + 'DFCN/' + scheme + next_fld + '/X_train2.npy')[:10000]
                Y_train = np.load('Features/' + 'DFCN/' + scheme + next_fld + '/Y_train2.npy')[:10000]
                X_val =   np.load('Features/' + 'DFCN/' + scheme + next_fld + '/X_val2.npy')[:1000]
                Y_val =   np.load('Features/' + 'DFCN/' + scheme + next_fld + '/Y_val2.npy')[:1000]
                X_test =  np.load('Features/' + 'DFCN/' + scheme + next_fld + '/X_test2.npy')[:1000]
                Y_test =  np.load('Features/' + 'DFCN/' + scheme + next_fld + '/Y_test2.npy')[:1000]

            if net in ['SVM/']:
                X_train = X_train[:5000]
                Y_train = Y_train[:5000]
                # X_val = X_val[:1000]
                # Y_val = Y_val[:1000]
                # X_test = X_test[:1000]
                # Y_test = Y_test[:1000]

            # Loading Models
            if net == 'SVM/':
                M1 = joblib.load('Models/'+net+'raw_model.joblib')
            else:
                M1 = models.load_model('Models/' + net + 'raw_model.h5')
                M1.load_weights('Models/' + net + 'raw_model_weights.h5')

            # training Models
            earlystoping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
            if net == 'Conv/':
                history = M1.fit(x=X_train, y=Y_train, batch_size=512, epochs=100, validation_data=(X_val, Y_val)
                                 , callbacks=[earlystoping])
            elif net == 'DFCN/':
                history = M1.fit(x=X_train, y=Y_train, batch_size=512, epochs=100, validation_data=(X_val, Y_val)
                                 , callbacks=[earlystoping])
            elif net == 'LSTM/':
                history = M1.fit(x=X_train, y=Y_train, batch_size=512, epochs=400, validation_data=(X_val, Y_val)
                                 , callbacks=[earlystoping])
            elif net == 'RBF/':
                history = M1.fit(x=X_train, y=Y_train, batch_size=512, epochs=100, validation_data=(X_val, Y_val)
                                 , callbacks=[earlystoping])
            elif net == 'SVM/':
                M1.fit(X_train, Y_train)
                joblib.dump(M1, 'Results/'+net+scheme+next_fld+'trained.joblib')

            # M1_trn_pred = M1.predict(X_train)
            M1_tst_pred = M1.predict(X_test)
            # M1_val_pred = M1.predict(X_val)

            # M1_trn_R2 = r2_score(Y_train, M1_trn_pred)
            M1_tst_R2 = r2_score(Y_test, M1_tst_pred)
            # M1_val_R2 = r2_score(Y_val, M1_val_pred)
            R2 = [0, 0, M1_tst_R2]
            R2_txt = 'R2-train = %f\nR2-val=%f\nR2-test=%f' % (R2[0], R2[1], R2[2])
            f = open('Results/' + net + scheme + next_fld + '/R2.txt', 'w')
            f.write(R2_txt)
            f.close()
            np.save('Results/' + net + scheme + next_fld + '/R2', R2)
            net_R2.append(M1_tst_R2)

            if net != 'SVM/':
                M1.save('Results/'+net+scheme+next_fld+'trained.h5')
                train_loss = history.history['loss']
                val_loss = history.history['val_loss']
                # train_mae = history.history['mae']
                # val_mae = history.history['val_mae']
                np.save('Results/' + net + scheme + next_fld + '/train_mse', train_loss)
                np.save('Results/' + net + scheme + next_fld + '/val_mse', val_loss)
                # np.save('Results/' + net + scheme + next_fld + '/train_mae', train_mae)
                # np.save('Results/' + net + scheme + next_fld + '/val_mae', val_mae)

                import matplotlib.pyplot as plt
                plt.plot(train_loss, color='blue', label='mse-train')
                plt.plot(val_loss, color='orange', label='mse-val')
                # plt.plot(train_mae, color='red', label='mae-train')
                # plt.plot(val_mae, color='green', label='mae-val')
                plt.xlabel('epochs')
                plt.ylabel('loss')
                plt.title(net[:-1] + ' && ' + scheme[2:])
                plt.legend()
                plt.savefig('Results/'+net + scheme + next_fld + '/loss.png', dpi=320)
                plt.show()
                del plt

            else:
                # temp1 = mse(y_true=Y_train, y_pred=M1_trn_pred)
                # temp2 = mse(y_true=Y_val, y_pred=M1_val_pred)
                temp3 = mse(y_true=Y_test, y_pred=M1_tst_pred)
                # temp4 = mae(y_true=Y_train, y_pred=M1_trn_pred)
                # temp5 = mae(y_true=Y_val, y_pred=M1_val_pred)
                temp6 = mae(y_true=Y_test, y_pred=M1_tst_pred)
                # temp7 = [temp1, temp2, temp3, temp4, temp5, temp6]
                temp7 = [temp3, temp6]
                np.save('Results/' + net + scheme + next_fld + '/LOSS', temp7)

                # L_txt = 'mse-train = %f\nmse-val=%f\nmse-test=%f\nmae-train=%f' \
                #         '\nmae-val=%f\nmae-test=%f' % (temp7[0], temp7[1], temp7[2], temp7[3], temp7[4], temp7[5])
                L_txt = 'mse-test = %f\nmae-test=%f'%(temp3, temp6)
                f = open('Results/' + net + scheme + next_fld + '/LOSS.txt', 'w')
                f.write(L_txt)
                f.close()

                print('Ending : ', scheme, '---', net[:-1], '---', next_fld[1:])

        scheme_R2.append(net_R2)

    scheme_R2 = np.array(scheme_R2)
    import matplotlib.pyplot as plt
    x = ['1', '3', '5', '7', '9', '11', '13', '15']
    for i in range(len(nets)):
        plt.plot(x, scheme_R2[i], label=nets[i][:-1], marker='*')
    plt.xlabel('time steps')
    plt.ylabel('adjusted R2')
    # plt.title(scheme[2:])
    plt.title('Simple sampling')
    plt.legend()
    # plt.savefig('Results/R2-curves/'+scheme[2:]+'.png')
    plt.savefig('Results/R2-curves/' + 'Origin' + '.png')
    plt.show()
    del plt
    # np.save('Results/R2-curves/'+scheme[2:], scheme_R2)
    np.save('Results/R2-curves/'+' Origin', scheme_R2)



