import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px
import random

def plot_missing_values(df, df_name):
    '''
    Plots mising values on interactive figure with plotly.
    Store in variable and call 
        `variable.show()`

    PARAMETERS
    ----------
        df: pandas dataframe
        df_name: string for title of dataframe

    RETURNS
    -------
        fig: figure
    '''
    missing_values = 100 * df.isna().sum()/len(df)
    missing_values = missing_values.reset_index()
    missing_values.columns = ['feature', '%_nan_values']

    fig = px.bar(missing_values, y='%_nan_values', x='feature',
             title=f'{df_name} Missing Values', template='ggplot2')
    return fig

def plot_state_sales(df, state):
    '''
    Plots monthly item sales for a state and it's stores.
    
    PARAMETERS
    ----------
        df: final dataframe (long form - unpivoted)
        state: string "CA", "TX", or "WI"
        
    RETURNS
    -------
        plots
    '''
    original_df = df[df.state_id==state].copy()
    store_ids = np.unique(original_df.store_id)
    num_stores = len(store_ids)
    
    original_df['date'] = pd.to_datetime(original_df['date'])
    state_df = original_df.groupby('date')['item_sales'].sum().reset_index()
    state_df = state_df.set_index('date')
    
    y = state_df['item_sales'].resample('MS').mean()
    
    fig, ax = plt.subplots(num_stores+1, figsize=(12, 18))
    y.plot(title=f'{state} Monthly Sales', ax=ax[0])
    
    for i, store in enumerate(store_ids):
        store_df = original_df[original_df.store_id == store].copy()
        store_df = store_df.groupby('date')['item_sales'].sum().reset_index()
        store_df = store_df.set_index('date')
        y = store_df['item_sales'].resample('MS').mean()
        
        r = random.random() 
        b = random.random() 
        g = random.random() 
  
        color = (r, g, b) 
        
        y.plot(title=f'{store} Monthly Sales', ax=ax[i+1], color=color);

def monthly_item_sales(df):
    '''
    Takes case sensitive user inputs to eventually display item_ids.
    Once user specifies an item_id, plots monthly sales.
    
    PARAMETERS
    ----------
        df: dataframe
    
    RETURNS
    -------
        plots item_id sales by month
    '''
    states = np.unique(df.state_id)
    state = input(f'Specify a state (case sensitive): {states} \n')
    
    while state is None:
        state = input(f'Specify a state (case sensitive): {states} \n')
    
    state_df = df[df.state_id==state].copy()
    stores = np.unique(state_df.store_id)
    store = input(f'Specify a store: {stores} \n')
    
    while store is None:
        store = input(f'Specify a store: {stores} \n')
    
    store_df = state_df[state_df.store_id==store].copy()
    cat_ids = np.unique(store_df.cat_id)
    category = input(f'Choose a department category to view items available\n{cat_ids} \n')
    
    while category is None:
        category = input(f'Choose a department category to view items available\n{cat_ids} \n')
    
    cat_df = store_df[store_df.cat_id==category].copy()
    items = np.unique(cat_df.item_id)
    
    for i, item in enumerate(items):
        print(item)
        if (i != 0) & (i % 5 == 0):
            decision = input('Display more items? Yes or No. \n')
            if decision=='Yes': pass
            elif decision=='No': break
            else: break
    
    item = input('Choose an item: \n')
    
    while item is None:
        item = input('Choose an item: \n')
    
    item_df = cat_df[cat_df.item_id==item].copy()
    
    item_df['date'] = pd.to_datetime(item_df['date'])
    item_df = item_df.groupby('date')['item_sales'].sum().reset_index()
    item_df = item_df.set_index('date')
    
    y = item_df['item_sales'].resample('MS').mean()
    
    r = random.random() 
    b = random.random() 
    g = random.random() 
    color = (r, g, b) 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    y.plot(title=f'{store} {item} Monthly Sales', color=color);