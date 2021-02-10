"""Main entry point into doing backtesting"""

import time

from sklearn.preprocessing import RobustScaler
import pandas as pd

from ExchangeInterface import ExchangeInterface
from TA import TA
from Trainer import Trainer
from Backtester import Backtester

if __name__ == '__main__':
    # Trading parameters
    exchange = 'coinbasepro'
    market = 'BTC/USD'
    fee = 0.001

    init_BTC = 1.0
    init_USD = 30000.0

    # key = os.environ["apiKey"]
    # secret = os.environ["apiSecret"]
    # passphrase = os.environ["passphrase"]

    # Get data
    exch_client = ExchangeInterface(exchange,
                                    market)
    start = 1546344000000 # Jan 1, 2019
    # start = 1514808000000 # Jan 1, 2018
    resp = exch_client.get_candles('1d', since=start)
    candles = resp
    while resp['timestamp'].iloc[-1] + 24*60*60*1000 < time.time()*1000:
        resp = exch_client.get_candles('1d', since=int(candles['timestamp'].iloc[-1]))
        candles = candles.append(resp, ignore_index=True)
    candles = candles.drop_duplicates(subset=['timestamp'], ignore_index=True)

    # Technical analysis
    ta_client = TA()
    features = ta_client.add_all_ta(candles)

    # Prepare model and data
    num_train = 300
    num_test = 200
    window_size = 10
    trainer = Trainer(features, window_size, fee, num_train, num_test)
    trainer.make_model()
    trainer.classes = trainer.label_data()

    # Scale data
    scaler = RobustScaler()
    scaler.fit(features.iloc[:num_train+num_test])
    trainer.feats_scaled = pd.DataFrame(scaler.transform(features), columns=features.columns, index=features.index)

    # Train model
    results = trainer.train_model()
    trainer.visualize_training_results(results)

    # Backtest model
    backtester = Backtester(
        candles.iloc[num_train+num_test:],
        trainer.feats_scaled[num_train+num_test:],
        trainer,
        fee,
        init_BTC=init_BTC,
        init_USD=init_USD,
        trade_size=0.01
    )

    tot_BTC, tot_USD = backtester.backtest()
    print('Init BTC: {:.4f}, Init USD: {:.2f}\nFinal BTC: {:.4f}, Final USD: {:.2f}'.format(init_BTC, init_USD, tot_BTC, tot_USD))

    # Run live
    # TODO: perhaps make live trading if time?

    print('done')
