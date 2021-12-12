import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from keras.optimizers import SGD, Adam
# from keras.metrics import MeanAbsoluteError as mae, MeanSquaredError as mse
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input, ZeroPadding2D, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, ConvLSTM2D, Flatten, LSTM
from keras.models import Sequential, Model
from keras.utils import plot_model, print_summary


# convolution model
def conv_model():

    lyr0 = Input(shape=(10, 10, 1))
    lyr = ZeroPadding2D(padding=(1, 1))(lyr0)
    lyr = Conv2D(filters=32, kernel_size=(2, 2), padding='same', strides=2, activation='relu')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)

    lyr = ZeroPadding2D(padding=(1, 1))(lyr)
    lyr = Conv2D(filters=64, kernel_size=(2, 2), padding='same', activation='relu')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)
    lyr = MaxPooling2D(pool_size=(2, 2))(lyr)

    lyr = ZeroPadding2D(padding=(1, 1))(lyr)
    lyr = Conv2D(filters=256, kernel_size=(2, 2), padding='same', activation='relu')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)
    lyr = MaxPooling2D(pool_size=(2, 2))(lyr)

    lyr = ZeroPadding2D(padding=(1, 1))(lyr)
    lyr = Conv2D(filters=30, kernel_size=(2, 2), padding='same', activation='relu')(lyr)
    # lyr = Dropout(rate=0.5)(lyr)
    lyr = MaxPooling2D(pool_size=(2, 2))(lyr)
    lyr = Flatten()(lyr)

    lyr = Dense(units=200, activation='relu')(lyr)
    lyr = Dense(units=20, activation='elu')(lyr)
    lyr_out = Dense(units=1, activation='linear')(lyr)

    M1 = Model(lyr0, lyr_out)
    M1.summary()

    M1.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    return M1

# fully connected model
def mlp_model():
    lyr0 = Input(shape=(10, ))
    lyr = Dense(units=50, activation='relu')(lyr0)
    lyr = Dense(units=17, activation='elu')(lyr)
    lyr = Dense(units=17, activation='elu')(lyr)
    lyr = Dense(units=7, activation='tanh')(lyr)
    lyr_out = Dense(units=1, activation='linear')(lyr)

    M1 = Model(lyr0, lyr_out)
    M1.summary()
    M1.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    return M1

def lstm_model():
    n_features = 1
    n_steps = 10
    M1 = Sequential()
    M1.add(LSTM(45, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    M1.add(LSTM(45, activation='relu'))
    M1.add(Dense(1))
    M1.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    M1. summary()
    return M1

class RBFNet:
    def __init__(self, n_bases = 5):
        self.n_bases = n_bases
        self.w = np.random.uniform(-1, +1, size=(1, self.n_bases))
    def get_weights(self, a):
        return self.w
    def set_weights(self, w_new):
        self.w = w_new

    def kmeans(self,X, k):
        """Performs k-means clustering for 1D input

        Arguments:
            X {ndarray} -- A Mx1 array of inputs
            k {int} -- Number of clusters

        Returns:
            ndarray -- A kx1 array of final cluster centers
        """

        # randomly select initial clusters from input data
        clusters = np.random.choice(np.squeeze(X), size=k)
        prevClusters = clusters.copy()
        stds = np.zeros(k)
        converged = False

        while not converged:
            """
            compute distances for each cluster center to each point 
            where (distances[i, j] represents the distance between the ith point and jth cluster)
            """
            distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))

            # find the cluster that's closest to each point
            closestCluster = np.argmin(distances, axis=1)

            # update clusters by taking the mean of all of the points assigned to that cluster
            for i in range(k):
                pointsForCluster = X[closestCluster == i]
                if len(pointsForCluster) > 0:
                    clusters[i] = np.mean(pointsForCluster, axis=0)

            # converge if clusters haven't moved
            converged = np.linalg.norm(clusters - prevClusters) < 1e-6
            prevClusters = clusters.copy()

        distances = np.squeeze(np.abs(X[:, np.newaxis] - clusters[np.newaxis, :]))
        closestCluster = np.argmin(distances, axis=1)

        clustersWithNoPoints = []
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) < 2:
                # keep track of clusters with no points or 1 point
                clustersWithNoPoints.append(i)
                continue
            else:
                stds[i] = np.std(X[closestCluster == i])

        # if there are clusters with 0 or 1 points, take the mean std of the other clusters
        if len(clustersWithNoPoints) > 0:
            pointsToAverage = []
            for i in range(k):
                if i not in clustersWithNoPoints:
                    pointsToAverage.append(X[closestCluster == i])
            pointsToAverage = np.concatenate(pointsToAverage).ravel()
            stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

        return clusters, stds

    def rbf_func(self, input, c, s):
        d = input - c
        return np.exp(-1 / (2 * s ** 2) * np.sum([ d[i] ** 2 for i in range(len(d)) ]) )

    # feedforward
    def __feedforward(self, input):
        x0 = input
        y1 = np.array([self.rbf_func(x0, c, s) for c, s, in zip(self.centers, self.stds)])
        y1 = np.reshape(y1, (1, self.n_bases))
        y_pred = np.dot(y1, self.w.T)[0]
        return y_pred[0], y1

    def fit(self, X_train, Y_train, val_split=0.1, val_data=None, epochs=50,
            lr=0.001, early_stopping=(True, 3), shuffle=False, verbose=True,
            optimzers = 'sgd', batch_size=100, time_series=True):

        # shuffling data
        if shuffle:
            X_temp = np.zeros_like(X_train)
            Y_temp = np.zeros_like(Y_train)
            rand_list = np.random.permutation(X_train.shape[0])
            for i in range(len(rand_list)):
                per_idx = rand_list[i]
                X_temp[i] = X_train[per_idx]
                Y_temp[i] = Y_train[per_idx]
            X_train = X_temp
            Y_train = Y_temp
            del X_temp
            del Y_temp

        # val data determination
        if val_data==None:
            X_val = X_train[:int(val_split*X_train.shape[0])]
            Y_val = Y_train[:int(val_split*X_train.shape[0])]
        else:
            X_val = val_data[0]
            Y_val = val_data[1]

        # computing clusters' center and std
        if time_series:
            self.centers, self.stds = self.kmeans(X=Y_train, k=self.n_bases)
        else:
            """
                This section must be implemented in future
            """
            print('[Error] FutureError: this section (time_series==False) currently has not been implemented')
            exit()

        # training section
        from sklearn.metrics import mean_squared_error as mse
        mse_train = []
        mse_val = []
        for i in range(epochs):
            for j in range(0, X_train.shape[0]):
                #for b in range(batch_size):
                x0 = X_train[j]
                y_true = Y_train[j]
                y_pred, y1 = self.__feedforward(input=x0)

                # backward
                # print('back')
                error = y_true - y_pred
                self.w = self.w - lr*(-1)*error*y1

            out_train = []
            for k in range(len(X_train)):
                x0 = X_train[k]
                y_pred, _ = self.__feedforward(input=x0)
                out_train.append(y_pred)

            out_val = []
            for k in range(len(X_val)):
                x0 = X_val[k]
                y_pred, _ = self.__feedforward(input=x0)
                out_val.append(y_pred)

            mse_train.append(mse(Y_train, out_train))
            mse_val.append(mse(Y_val, out_val))
            print('[epoch %d] mse train: %f, mse val: %f' %(i + 1, mse_train[-1], mse_val[-1]))

            if early_stopping[0] and i>early_stopping[1]:
                if mse_val[-1]>mse_val[-1*early_stopping[1]]:
                    print('---------------\nEarly stopping activated')
                    break
        return mse_train, mse_val

    def predict(self, X):
        preds = []
        for x0 in X:
            y1 = np.array([self.rbf_func(x0, c, s) for c, s, in zip(self.centers, self.stds)])
            y1 = np.reshape(y1, (1, self.n_bases))
            y_pred = np.dot(y1, self.w.T)[0]
            preds.append(y_pred[0])
        return np.array(preds)

def RBFNet2():
    lyr0 = Input(shape=(10,))
    lyr = Dense(units=50, activation='tanh')(lyr0)
    lyr_out = Dense(units=1, activation='linear')(lyr)
    M1 = Model(lyr0, lyr_out)
    M1.summary()
    M1.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    return M1

M1 = lstm_model()

