import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
import random
import datetime
import statsmodels.tsa.api as smt

#####################################################
##               USED IN 0_eda.ipynb               ##
#####################################################

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

#####################################################
##            USED IN 1_regression.ipynb           ##
#####################################################

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

def train_test_spli(df):
    '''
    Returns train and test/holdout data ready for model training and evaluation.
    Different than regular train_test_split for time series specific problems.
        Cannot be randomized bc data is sequential.
    
    PARAMETERS
    ----------
        df: dataframe with monthly entries
    
    RETURNS
        train: series, training dataframe
        test: series, last 12 months for holdout/testing
    '''
    df = df.drop(['date', 'item_sales'],axis=1)
    train, test = df[0:-12].values, df[-12:].values
    
    return train, test

def scale_data(train_set, test_set):
    # apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    
    # y is the first column `sales_diff`
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler

def load_data(state):
    '''
    Loads and cleans original dataframes for plotting.
    
    PARAMETERS
    ----------
        state: string, 'ca', 'tx', 'wi'
        
    RETURNS
    -------
        dataframe with date and item sales only
    '''
    df = pd.read_csv(f'../data/{state}_supervised.csv')
    subset = df[['date', 'item_sales']].copy()
    
    # just some cleaning for plots
    subset.date = subset.date.apply(lambda x: str(x)[:-3])
    subset = subset.groupby('date')['item_sales'].sum().reset_index()
    subset.date = pd.to_datetime(subset.date)
    
    return subset

def plot_results(state, model, original_df, results_df):
    '''
    Plots original item_sales with predicted sales
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(original_df.date, original_df.item_sales, data=original_df, label='Original Data', ax=ax, color=gen_random_color())
    sns.lineplot(results_df.date, results_df.pred_value, data=results_df, label='Predictions', ax=ax, color=gen_random_color());
    ax.set(xlabel = "Date", ylabel = "Sales", title = f"{model.__class__.__name__} {state} Sales Forecasting Prediction")
    plt.savefig(f'../images/{state}_{model.__class__.__name__}_forecast.png')
    
def scoring(state, model, original_df, results_df):
    '''
    Prints RMSE & R2 scores, saves to csv
    '''
    model_scores = {}

    rmse = np.sqrt(mean_squared_error(original_df.item_sales[-12:], results_df.pred_value[-12:]))
    r2 = r2_score(original_df.item_sales[-12:], results_df.pred_value[-12:])
    
    scores = pd.read_csv('../data/model_scores.csv')
    
    scores[f'{model.__class__.__name__}_{state}'] = [rmse, r2]

    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")
    pd.DataFrame(scores).to_csv(f'../data/model_scores.csv', index=False)

def model(state, regressor):
    '''
    Loads data, performs a train test split then scales each split.
    Trains a regressor, makes predicted sales difference,
    adds sales difference to actual item sales to get 
    month's forecasted sales.
    
    Lastly, plots model results and saves RMSE & R2 scores.
    
    PARAMETERS
    ----------
        state: 'ca', 'tx', 'wi'
        regressor: untrained model object
        scaler: fit scaler object
    
    RETURNS
    -------
        plots forecasted prediction on actual item_sales values
        saves model scores to csv in data folder
    '''
    state_df = pd.read_csv(f'../data/{state}_supervised.csv')
    train, test = train_test_spli(state_df)
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)
    
    model = regressor
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # y_pred = 12 predicted, scaled, sales_differ values
    # y_pred.shape = (12,)
    # reshape y_pred and X_test to concatenate
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    pred_test_set = []
    for i in range(0, len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[i], X_test[i]], axis=1))
        
    # pred_test_set.shape = (12, 1, 13)
    # reshape predictions back to (12, 13)
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    pred_test_set_inverted = scaler_object.inverse_transform(pred_test_set)
    
    # load original dataframe to add predicted sales_differ to previous
    # month values to get actual item sales prediction
    df = load_data(state)
    
    results = []
    dates = list(df[-13:].date)
    item_sales = list(df[-13:].item_sales)
    
    # predicting the nth difference in item sales:
    # need to add the prediction to the month previous sales
    for i in range(0, len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[i][0] + item_sales[i])
        result_dict['date'] = dates[i + 1]
        results.append(result_dict)
    
    results_df = pd.DataFrame(results)
    
    plot_results(state, model, df, results_df)
    
    scoring(state, model, df, results_df)