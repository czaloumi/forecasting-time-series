import pandas as pd
import numpy as np
import random
import datetime
import requests
from helper_functions import *

def load_separate_datasets():
    '''
    Download data from: https://www.kaggle.com/c/m5-forecasting-accuracy/data
    Move into data folder.
    '''
    train_sales = pd.read_csv("../data/sales_train_evaluation.csv")
    calendar = pd.read_csv("../data/calendar.csv")
    sell_prices = pd.read_csv("../data/sell_prices.csv")
    return train_sales, calendar, sell_prices

def clean_melt_and_merge(train_sales, calendar, sell_prices):
    '''
    Minor cleaning, melts train_sales, and merges all 3.
    '''
    to_drop = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    calendar.drop(to_drop, axis=1, inplace=True)

    train_sales_melt = pd.melt(train_sales, 
            id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
            var_name='d', value_name='item_sales')

    df = pd.merge(train_sales_melt, calendar, on='d')

    to_drop = ['weekday', 'wday', 'month', 'year']
    df.drop(to_drop, axis=1, inplace=True)

    final = df.merge(sell_prices, on=['item_id', 'store_id', 'wm_yr_wk'], how='left')
    return final

def save(df):
    df.to_csv('../data/final_df.csv')

if __name__ == "__main__":
    train_sales, calendar, sell_prices = load_separate_datasets()

    df = clean_melt_and_merge(train_sales, calendar, sell_prices)
    print(df.head())

    save(df)