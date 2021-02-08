"""Provides the trainer client."""
import numpy as np


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

        for ii in range(len(self.feats) - 1):
            open_price = self.feats['open'][ii + 1]
            close_price = self.feats['close'][ii + 1]
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
        Bruh literally what is this supposed to do
        :param x: a 1D numpy array
        :return y: a 2D numpy array
        """
        y = np.zeros((x.size, x.max() + 1))
        y[np.arange(x.size), x] = 1

        return y
