import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import plotly.express as px

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