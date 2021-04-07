import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

dirpath_1 = Path('.../top_events/')
dirpath_2 = Path('.../handpicks_events/')


# Aggregate all metrics
def get_frames(dirpath):
    for file in dirpath.iterdir():
        df = pd.read_csv(file, index_col=0, header=None).T
        df = df.applymap(lambda x: x.strip('\t'))
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df.set_index('Event', drop=True, inplace=True)
        df.index.name = 'DateTime'
        yield df


def get_set():
    df_1, df_2, df_3, df_4 = get_frames(dirpath_1)

    for frame in [df_2, df_3, df_4]:
        frame.columns = df_1.columns.to_list()

    df = pd.concat([df_1, df_2, df_3, df_4], axis=0)
    df.sort_index(inplace=True)
    df = df.apply(lambda x: pd.to_numeric(x))

    df_1, df_2, df_3, df_4 = get_frames(dirpath_2)

    for frame in [df_2, df_3, df_4]:
        frame.columns = df_1.columns.to_list()

    df_others = pd.concat([df_1, df_2, df_3, df_4], axis=0)
    df_others.sort_index(inplace=True)
    df_others = df_others.apply(lambda x: pd.to_numeric(x))

    # Deal with duplicates metrics
    df_e = pd.concat([df, df_others], axis=1).sort_index(axis=1)
    df = df_e.iloc[:, ~df_e.columns.duplicated()]
    return df


def get_autoregression(frame, metric, window, lags):
    # train-test split
    X = frame[f'{metric}'].values
    train, test = X[1:X.size - 100], X[X.size - 100:]

    # train autoregression
    window = window
    model = AutoReg(train, lags=lags)
    model_fit = model.fit()
    coef = model_fit.params

    # Walk forward over time steps in test
    history = train[train.size - window:]
    history = [history[i] for i in range(history.size)]
    preds = list()

    for t in range(test.size):
        length = len(history)
        lag = [history[i] for i in range(length - window, length)]
        yhat = coef[0]

        for d in range(window):
            yhat += coef[d + 1] * lag[window - d - 1]

        obs = test[t]
        preds.append(yhat)
        history.append(obs)
        print('predicted:', yhat, 'expected', obs)

    rmse = sqrt(mean_squared_error(test, preds))
    print('Test RMSE:', rmse)

    # plot the results
    # plt.plot(test, label='Actual Observations')
    # plt.plot(preds, color='pink', label='Prediction')
    # plt.legend(loc="best")
    # plt.show()

    return None


def main():
    data = get_set()

    print('RMSE daily data:')
    get_autoregression(frame=data,
                       metric='Tap_Challenge_Accomplished_This_Week',
                       window=7, lags=150)

    print('RMSE weekly resample:')
    get_autoregression(frame=data.resample('W').median(),
                       metric='Tap_Challenge_Accomplished_This_Week',
                       window=7, lags=30)


if __name__ == '__main__':
    main()
