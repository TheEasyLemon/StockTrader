"""Provides TA, a wrapper class for technical analysis."""

import ta


class TA:
    def __init__(self):
        pass

    def add_all_ta(self, df):
        """
        Returns a dataframe that has ta methods.
        :param df: a pandas dataframe that contains open, high, low, close, and volume series.
        :return: a TA dataframe that contains ta methods.
        """
        df_ta = ta.add_all_ta_features(df, open='open', high='high', low='low', close='close', volume='volume',
                                       fillna=True)
        return df_ta
