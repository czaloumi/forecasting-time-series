**********************************************
# Wallmart Sales Forecasting
**********************************************

#### Author: Chelsea Zaloumis

*Last update: 3/19/2021*

![title](images/wallmart.jpeg)

# Background
---
Based off the M5 Forecasting Kaggle Competition: https://www.kaggle.com/c/m5-forecasting-accuracy/overview

*Can you estimate, as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart?*

This project helped me better understand time series and forecasting problems.

# Exploration
---
Original data can be found at the kaggle link to the competition above. Data contains item sales by state.

 <p align="center">
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/monthly_sales_state.png" width="75%" height="75%"/>
 </p>

For exploring the data, users can find functions in `src/helper_functions.py` and example usage in `src/0_eda.ipynb`. Examples below for drilling down to each state's various stores, departments, and items.

<p align="center">
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/ca_monthly_sales.png" width="75%" height="75%"/>
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/tx_monthly_sales.png" width="75%" height="75%"/>
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/wi_monthly_sales.png" width="75%" height="75%"/>
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/dept_ex_sales.png" width="75%" height="75%"/>
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/item_ex_sales.png" width="75%" height="75%"/>
 </p>

# Initial Modeling
---
Compared three regression models out of box on forecasting California's monthly sales. Linear Regression outperformed both Random Forest and Gradient Boosting.
 <p align="center">
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/oob_regr_comparison.png" width="75%" height="75%"/>
 <img src="https://github.com/czaloumi/forecasting-time-series/blob/main/images/ca_LinearRegression_forecast.png" width="75%" height="75%"/>
 </p>
