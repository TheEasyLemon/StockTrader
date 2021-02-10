"""Provides backtesting capability."""
import numpy as np
from datetime import datetime


class Backtester:
    def __init__(self, data, data_scaled, trainer, fee, init_BTC, init_USD, trade_size):
        self.data = data
        self.data_scaled = data_scaled
        self.trainer = trainer
        self.fee = fee

        self.init_BTC = init_BTC
        self.init_USD = init_USD
        self.trade_size = trade_size

    def backtest(self):
        tot_BTC = self.init_BTC
        tot_USD = self.init_USD

        action_names = ['hold', 'buy', 'sell']

        # Go through all data and run strategy at every data point
        for ii in range(self.trainer.window_size,len(self.data)-1):
            data_window = self.data_scaled.iloc[ii-self.trainer.window_size:ii]
            data_array = np.array(data_window)
            data_tensor = np.reshape(data_array, (1, self.trainer.window_size, self.trainer.num_feats))
            y = self.trainer.model.predict(data_tensor)
            best_action = np.argmax(y)

            # [0.2 0.9 0.6]

            if best_action == 1: # buy, then sell
                price_buy = self.data['open'].iloc[ii]
                price_sell = self.data['close'].iloc[ii]

                # Place buy
                tot_BTC += self.trade_size
                tot_USD -= self.trade_size*price_buy*(1+self.fee)

                # Place sell
                tot_BTC -= self.trade_size
                tot_USD += self.trade_size*price_sell*(1-self.fee)
            elif best_action == 2: # sell, then buy
                price_sell = self.data['open'].iloc[ii]
                price_buy = self.data['close'].iloc[ii]

                # Place sell
                tot_BTC -= self.trade_size
                tot_USD += self.trade_size * price_sell * (1 - self.fee)

                # Place buy
                tot_BTC += self.trade_size
                tot_USD -= self.trade_size * price_buy * (1 + self.fee)
            else: # hold
                pass

            t = self.data['timestamp'].iloc[ii]
            print('Backtest: {} - {}'.format(datetime.utcfromtimestamp(t/1000), action_names[best_action]))

        return tot_BTC, tot_USD