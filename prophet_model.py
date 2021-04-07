import pandas as pd
from scipy import stats
from helpers.process import get_sub, get_dfs, make_comparison_dataframe, calculate_forecast_errors, inverse_boxcox
from prophet import Prophet

# Import the data
df, act, quizz = get_dfs()

# Format the data
sub_a = get_sub(act, from_date='09/2019')
sub_q = get_sub(quizz, from_date='09/2019')

prediction_size = 30
train_df = sub_a[:-prediction_size]
train_df.tail(n=3)

# Baseline model
m = Prophet(yearly_seasonality=False, daily_seasonality=True)
m.fit(train_df)

future = m.make_future_dataframe(periods=prediction_size)
future.tail(n=3)

forecast = m.predict(future)
cmp_df = make_comparison_dataframe(sub_a, forecast)

for err_name, err_value in calculate_forecast_errors(cmp_df, prediction_size).items():
    print(err_name, err_value)

# BoxCox transformation
train_df2 = train_df.copy().set_index("ds")
train_df2["y"], lambda_prophet = stats.boxcox(train_df2["y"])
train_df2.reset_index(inplace=True)

# Optimized model
m2 = Prophet(yearly_seasonality=False, daily_seasonality=True)
m2.fit(train_df2)
future2 = m2.make_future_dataframe(periods=prediction_size)
forecast2 = m2.predict(future2)

for column in ["yhat", "yhat_lower", "yhat_upper"]:
    forecast2[column] = inverse_boxcox(forecast2[column], lambda_prophet)

cmp_df2 = make_comparison_dataframe(sub_a, forecast2)
for err_name, err_value in calculate_forecast_errors(cmp_df2, prediction_size).items():
    print(err_name, err_value)


