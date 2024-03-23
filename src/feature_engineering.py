import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DeltaPrices(BaseEstimator, TransformerMixin):
    def __init__(self, auction = True): 
        self.auction = auction
    
    def transform(self, df):
        df = df.eval("delta_bid_ask = (bid_price - ask_price) / (bid_price + ask_price)")
        df = df.eval("delta_wap_ref = (wap - reference_price) / (wap + reference_price)")
        df = df.eval("delta_bid_ask_size = (bid_size - ask_size) / (bid_size + ask_size + 1)")
        df = df.eval("ratio_bid_ask_matched_size = (bid_size - ask_size) / (matched_size + 1)")
        df = df.eval("imbalance_signed = imbalance_buy_sell_flag * imbalance_size")
        df = df.eval("delta_bid_ref = (bid_price - reference_price) / (bid_price + reference_price)")
        df = df.eval("delta_ask_ref = (ask_price - reference_price) / (ask_price + reference_price)")
        df = df.eval("delta_ask_wap = (ask_price - wap) / (ask_price + wap)")
        df = df.eval("delta_bid_wap = (bid_price - wap) / (bid_price + wap)")
        df = df.eval("imbalance_per_delta_bidask_price = (imbalance_signed) * (bid_price - ask_price)")
        df = df.eval("delta_imbalance_matched = (imbalance_signed - matched_size)/(matched_size + imbalance_signed)")
        df = df.eval("ratio_bid_ask_size = bid_size / ask_size")


        if self.auction:
            df = df.eval("delta_near_far = (near_price - far_price) / (near_price + far_price)")
            df = df.eval("delta_far_ref = (far_price - reference_price) / (far_price + reference_price)")
            df = df.eval("delta_near_wap = (near_price - wap) / (near_price + wap)")
            df = df.eval("delta_near_ref = (near_price - reference_price) / (near_price + reference_price)")
            df = df.eval("delta_near_far_on_matched = (near_price - far_price) / (matched_size + 1)* 10000") #26/10

        return df

class AverageWap(BaseEstimator, TransformerMixin):

    def __init__(self, wap=True): 
        self.wap = wap
    
    def transform(self, df):
        if self.wap:
            def compute_w_a_wap(wap, ask_size, bid_size):
                return (wap * (bid_size + ask_size)).sum() / (bid_size + ask_size).sum()

            _ = df.groupby(['date_id', 'seconds_in_bucket'])\
                .apply(lambda x : compute_w_a_wap(x.wap, x.ask_size, x.bid_size))\
                .reset_index().rename(columns = {0 : 'w_a_wap'})
            df = df.merge(_, on = ['date_id', 'seconds_in_bucket'], validate = 'm:1')\
                .assign(wap_less_wawap = lambda df_ : (df_.wap - df_.w_a_wap) * 10000)
        return df

class Aggregates(BaseEstimator, TransformerMixin):
    def __init__(self, cols=["wap", "ask_size", "bid_size"], funcs=["mean", "std"]):
        self.cols = cols
        self.funcs = funcs

    def transform(self, df):
        for col in self.cols:
            for func in self.funcs:
                new_col = f"{col}_agg_difference_{func}"
                df[new_col] = df[col] - df.groupby(["stock_id", "seconds_in_bucket"])[col].transform(func)
        return df

class RollingFeatEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, cols=["wap"], funcs=["mean", "std", "min", "max", "median"]):
        self.cols = cols
        self.funcs = funcs

    def transform(self, df):
        new_cols = [f"{col}_rolling_{func}" for col in self.cols for func in self.funcs]
        df[new_cols] = df.groupby(["stock_id", "date_id"])[self.cols].rolling(2).aggregate(self.funcs).droplevel([0,1])

        return df


class EWMFeatEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, cols=["wap"], cols_std=["reference_price", "matched_size"], span=4):
        self.cols = cols
        self.cols_std = cols_std
        self.span = span

    def transform(self, df):
        new_cols = [f"{col}_ewm_mean" for col in self.cols]
        df[new_cols] = df.groupby(["stock_id"], as_index=False)[self.cols].transform(lambda x: x.ewm(span=self.span).mean())
        
        new_cols = [f"{col}_ewm_std" for col in self.cols]
        df[new_cols] = df.groupby(["stock_id"], as_index=False)[self.cols].transform(lambda x: x.ewm(span=self.span).std())

        return df

class MACD(BaseEstimator, TransformerMixin):
    def __init__(self, cols=["wap"]):
        self.cols = cols
    def transform(self, df):
        new_cols = [f"{col}_macd" for col in self.cols]
        df[new_cols] = df.groupby(["stock_id"])[self.cols].transform(lambda x: x.ewm(span=4).mean()) - df.groupby(["stock_id"])[self.cols].transform(lambda x: x.ewm(span=12).mean())
        return df


def run(df):
    macd = MACD()
    ewm = EWMFeatEngineering()
    rolling = RollingFeatEngineering()
    delta = DeltaPrices()
    average_wap = AverageWap()
    aggregates = Aggregates()
    
    pipeline = Pipeline([
        ('delta', delta),
        ('average_wap', average_wap),
        ('aggregates', aggregates),
        ('macd', macd),
        ('ewm', ewm),
        ('rolling', rolling)
    
        
    ])

    df = pipeline.transform(df)
    return df