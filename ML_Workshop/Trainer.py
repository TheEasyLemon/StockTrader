"""Provides the trainer client."""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
plt.style.use('bmh') # https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html


class Trainer:
    def __init__(self, features, window_size, fee, num_train, num_test):
        self.feats = features
        self.window_size = window_size
        self.fee = fee
        self.num_train = num_train
        self.num_test = num_test
        self.num_feats = features.shape[1]
        self.classes = None
        self.feats_scaled = None
        self.model = None

    def label_data(self):
        """
        If the open price of the next period is lower than than closing price (accounting for fees),
        then we want to buy at the open and sell at the close. If vice versa, then we want to sell at
        the open and buy at the close. Otherwise, we want to do nothing.

        We do this by labelling each time period. We label 1 for buy at open/sell at close,
        2 for sell at open/buy at close, and 0 for do nothing.

        The fees are calculated as such: whenever you sell, you sell some percent under market
        price (a "spread", so that platforms make money), and buy some percent above market
        price. We adjust our profits accordingly according to a buy/sell multiplier.
        :return labels:
        """
        labels = []

        for i in range(len(self.feats) - 1):
            open_price = self.feats['open'][i + 1]
            close_price = self.feats['close'][i + 1]
            buy_fee_multiplier = 1 + self.fee
            sell_fee_multiplier = 1 - self.fee

            if close_price * sell_fee_multiplier - open_price * buy_fee_multiplier > 0:
                labels.append(1)
            elif open_price * sell_fee_multiplier - close_price * buy_fee_multiplier > 0:
                labels.append(2)
            else:
                labels.append(0)

        return labels

    def one_hot_encode(self, x):
        """
        For every element i in x, create a row vector x.max() long and
        have the ith element in that vector be one, with the rest zeros.
        Horizontal concatenate for a 2D matrix.

        [ 0 1 2 ] -> [ 1 0 0 ;
                       0 1 0 ;
                       0 0 1 ]

        :param x: a 1D numpy array
        :return y: a 2D numpy array
        """
        y = np.zeros((x.size, x.max() + 1))
        y[np.arange(x.size), x] = 1

        return y

    def make_model(self):
        # This creates the LSTM model.  Lots of room for innovation here.

        """
        Tensorflow is a platform for machine learning. It provides many models that we can use.
        Sequential groups a linear stack of layers into a tk.keras.Model, the primary object that
        stores the model.
        """
        self.model = Sequential()

        self.model.add(LSTM(
            units=100,
            activation='tanh',
            input_shape=(self.window_size, self.num_feats)))
        self.model.add(Dense(units=3,
                             activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=0.001),
                           metrics=['accuracy'])
        self.model.summary()

        return self.model

    def split_data(self):
        train_X = self.feats_scaled.iloc[:self.num_train]
        train_y = self.classes[:self.num_train]
        test_X = self.feats_scaled.iloc[self.num_train:self.num_train + self.num_test]
        test_y = self.classes[self.num_train:self.num_train + self.num_test]

        return train_X, train_y, test_X, test_y

    def format_data(self, X):
        # Creating a 3D array to hold all the data samples
        X_tensor = np.zeros((X.shape[0] - self.window_size, self.window_size, self.num_feats))
        for ii in range(X_tensor.shape[0] - self.window_size):
            X_tensor[ii, :, :] = X[ii:ii + self.window_size, :]

        return X_tensor

    def train_model(self):
        train_X, train_y, test_X, test_y = self.split_data()

        # Training data
        train_X = np.array(train_X)
        train_X = self.format_data(train_X)
        train_y = np.array(train_y[self.window_size:])
        train_y = self.one_hot_encode(train_y)

        # Testing data
        test_X = np.array(test_X)
        test_X = self.format_data(test_X)
        test_y = np.array(test_y[self.window_size:])
        test_y = self.one_hot_encode(test_y)

        # Train the model
        training_results = self.model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=10, batch_size=1)

        return training_results

    def visualize_training_results(self, results):
        history = results.history
        plt.figure(figsize=(10, 5))
        plt.plot(history['val_loss'])
        plt.plot(history['loss'])
        plt.legend(['val_loss', 'loss'])
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(history['val_accuracy'])
        plt.plot(history['accuracy'])
        plt.legend(['val_accuracy', 'accuracy'])
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()
