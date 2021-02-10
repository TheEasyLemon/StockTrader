"""
Provides ExchangeInterface. Currently is able to trade on Binance and Coinbase Pro.
"""
import pandas as pd
import ccxt


class ExchangeInterface:
    def __init__(self, exchange, market, key=None, secret=None, phrase=None):
        """
        Interface that uses ccxt to interface with cryptocurrency markets

        :param exchange: a string containing cryptocurrency exchange ('binance', 'coinbasepro')
        :param market: a string containing the base cryptocurrency, followed by a slash,
        followed by the quote ('BTC/USD')
        :param key: key to access your account
        :param secret: secret to access your account
        :param phrase: phrase to access your account
        """
        # Setting up CCXT client
        exchange_class = getattr(ccxt, exchange)

        if exchange == 'coinbasepro':
            if key:
                self.client = exchange_class({
                    'apiKey': key,
                    'secret': secret,
                    'password': phrase,
                    'enableRateLimit': True,
                    'rateLimit': 400
                })
            else:
                self.client = exchange_class({
                    'enableRateLimit': True,
                    'rateLimit': 400
                })
        elif exchange == 'binance':
            if key:
                self.client = exchange_class({
                    'apiKey': key,
                    'secret': secret,
                    'nonce': ccxt.Exchange.milliseconds,
                    'enableRateLimit': True,
                    'rateLimit': 200
                })
            else:
                self.client = exchange_class({
                    'nonce': ccxt.Exchange.milliseconds,
                    'enableRateLimit': True,
                    'rateLimit': 200
                })
        else:
            raise Exception(f"{exchange} is not a supported exchange.")

        self.client.load_markets()

        self.exchange = exchange
        self.market = market
        market_split = self.market.split('/')
        self.base = market_split[0] # for example, BTC
        self.quote = market_split[1] # for example, USD

    def get_latest_price(self):
        tick = self.client.fetch_ticker(self.market)
        return tick['last']

    def get_candles(self, res, since=None):
        if since:
            resp = self.client.fetch_ohlcv(self.market, res, since=since)
        else:
            resp = self.client.fetch_ohlcv(self.market, res)
        df = pd.DataFrame(resp, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
