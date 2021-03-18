import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
import random
import datetime
import statsmodels.tsa.api as smt

def gen_random_color():
    r = random.random() 
    b = random.random() 
    g = random.random() 
  
    return (r, g, b)

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
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f'{state} Monthly Sales')
    ax.plot(y, color=gen_random_color(), label=f'{state}')
    
    for i, store in enumerate(store_ids):
        store_df = original_df[original_df.store_id == store].copy()
        store_df = store_df.groupby('date')['item_sales'].sum().reset_index()
        store_df = store_df.set_index('date')
        y = store_df['item_sales'].resample('MS').mean()
        
        ax.plot(y, color=gen_random_color(), label=f'{store}')
    
    ax.legend();
    
def compare_state_sales(df):
    '''
    Plot sales per month for each state
    
    PARAMETERS
    ----------
        df: dataframe
    
    RETURNS
    -------
        plots sales by month
    '''
    num_states = len(np.unique(df.state_id))
        
    fig, ax = plt.subplots(figsize=(12, 6))
        
    for i, state in enumerate(np.unique(df.state_id)):
        state_df = df[df.state_id == state].copy()
        state_df['date'] = pd.to_datetime(state_df['date'])
        state_df = state_df.groupby('date')['item_sales'].sum().reset_index()
        state_df = state_df.set_index('date')

        y = state_df['item_sales'].resample('MS').mean()

        ax.plot(y, color=gen_random_color(), label=f'{state}')
            
    ax.set_title('Monthly Sales By State')
    ax.legend();
    
def monthly_item_sales(df):
    '''
    Takes case sensitive user inputs to eventually display item_ids.
    If user specifies a department, function will plot department monthly sales.
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
    
    yay_or_nay = input(f'Plot department sales? Yes or No. \n')
    
    if yay_or_nay == 'Yes':
        cat_df['date'] = pd.to_datetime(cat_df['date'])
        cat_df = cat_df.groupby('date')['item_sales'].sum().reset_index()
        cat_df = cat_df.set_index('date')
    
        y = cat_df['item_sales'].resample('MS').mean()
    
        fig, ax = plt.subplots(figsize=(12, 6))
        y.plot(title=f'{store} {category} Monthly Sales', color=gen_random_color());
    
    else:
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

        fig, ax = plt.subplots(figsize=(12, 6))
        y.plot(title=f'{store} {item} Monthly Sales', color=gen_random_color());
        
def monthly_sales(data):
    '''
    Function returns a df with two columns:
        date: datetime
        sales: total item sales (not $)
    '''
    monthly_data = data.copy()
    monthly_data.date = pd.to_datetime(monthly_data.date)
    monthly_data = monthly_data.groupby('date')['item_sales'].sum().reset_index()
    
    return monthly_data

def time_plot(data, x_col, y_col, title):
    '''
    Plots monthly sales with average for comparison.
    Users can view if the data is stationary or not based on average.
    
    PARAMETERS
    ----------
        data: dataframe
        x_col: datetime
        y_col: value to observe
        title: string
        
    RETURNS
    -------
        plot
    '''
    first = data.set_index('date')
    
    y = pd.DataFrame(first[y_col].resample('MS').mean())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y, color=gen_random_color(), label='Total Sales')
    
    second = data.groupby(data.date.dt.year)[y_col].mean().reset_index()
    second.date = pd.to_datetime(second.date, format='%Y')
    sns.lineplot((second.date + datetime.timedelta(6*365/12)), y_col, data=second, ax=ax, color='red', label='Mean Sales')   
    
    ax.set(xlabel = "Date",
           ylabel = "Sales",
           title = title)
    
    sns.despine()
    
def plot_shared_yscales(axs, x, ys, titles):
    ymiddles =  [ (y.max()+y.min())/2 for y in ys ]
    yrange = np.max([ (y.max()-y.min())/2 for y in ys ])
    for ax, y, title, ymiddle in zip(axs, ys, titles, ymiddles):
        ax.plot(x, y)
        ax.set_title(title)
        ax.set_ylim((ymiddle-yrange, ymiddle+yrange))
        
def plot_seasonal_decomposition(axs, series, sd):
    plot_shared_yscales(axs,
                        series.index,
                        [series.item_sales, sd.trend, sd.seasonal, sd.resid],
                        ["Raw Series", 
                         "Trend Component $T_t$", 
                         "Seasonal Component $S_t$",
                         "Residual Component $R_t$"])
    
def correlation_plots(data, lags=None):
    '''
    Plots stationary series, autocorrelation, and partial autocorrelation.
    
    PARAMETERS
    ----------
        data: df
    
    RETURNS
    -------
        3 subplot figure
    '''
    dt_data = data.set_index('date').drop('item_sales', axis=1)
    dt_data.dropna(axis=0)
    
    y = pd.DataFrame(dt_data['sales_diff'].resample('MS').mean())
    
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))
    
    y.plot(ax=raw, figsize=(12, 6), color=gen_random_color())
    smt.graphics.plot_acf(dt_data, lags=lags, ax=acf, color=gen_random_color())
    smt.graphics.plot_pacf(dt_data, lags=lags, ax=pacf, color=gen_random_color())
    sns.despine()
    plt.tight_layout()

def get_diff(data, state):
    data['sales_diff'] = data.item_sales.diff()
    data = data.dropna()
    return data

def clean_df(df, state):
    '''
    Simplifies problem from daily to monthly
    '''
    df = df.resample('MS').mean().copy()
    df = df.reset_index()
    return(get_diff(df, state))

def generate_supervised(df, state, window_width):
    '''
    Creates a dataframe for transformation from time series to supervised.
    
    PARAMETERS
    ----------
        df: dataframe to add lag columns to
        state: string, 'ca', 'tx', or 'wi'
        window_width: int for seasonal aspect (365 for days, 12 for months)
    
    RETURNS
    -------
        dataframe for supervised learning
    '''
    supervised_df = df.copy()
    
    #create column for each lag
    for i in range(1,window_width+1):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)
    
    #drop null values
    supervised_df = supervised_df.dropna().reset_index(drop=True)
    
    supervised_df.to_csv(f'../data/{state}_supervised.csv', index=False)
    
    return supervised_df