import pandas as pd
import numpy as np
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_squared_log_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import TimeSeriesSplit


def get_dfs():
    df = (pd.read_csv('.../all_metrics.csv',
                      parse_dates=['DateTime'],
                      index_col='DateTime',
                      usecols=["DateTime", 'Tap_Challenge_Accomplished_This_Week', 'Answer_Quizz_Question']))

    quizz = df.drop('Tap_Challenge_Accomplished_This_Week', axis=1).loc['2020-07-20	':]
    act = df.drop('Answer_Quizz_Question', axis=1).loc['2020-07-20':]
    act.columns = ['Actions']
    quizz.columns = ['Answers']
    return df, act, quizz


def make_comparison_dataframe(historical, forecast):
    """Join the history with the forecast.

       The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
    """
    return forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]].join(
        historical.set_index("ds")
    )


def inverse_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)


def calculate_forecast_errors(df, prediction_size):
    """Calculate MAPE and MAE of the forecast.

       Args:
           df: joined dataset with 'y' and 'yhat' columns.
           prediction_size: number of days at the end to predict.
    """

    # Make a copy
    df = df.copy()

    # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
    df["e"] = df["y"] - df["yhat"]
    df["p"] = 100 * df["e"] / df["y"]

    # Recall that we held out the values of the last `prediction_size` days
    # in order to predict them and measure the quality of the model.

    # Now cut out the part of the data which we made our prediction for.
    predicted_part = df[-prediction_size:]

    # Define the function that averages absolute error values over the predicted part.
    error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

    # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
    return {"MAPE": error_mean("p"), "MAE": error_mean("e")}


def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):
    """
        Returns error on CV

        params - vector of parameters for optimization
        series - dataset with timeseries
        slen - season length for Holt-Winters model
    """
    # errors array
    errors = []

    values = series.values
    alpha, beta, gamma = params

    # set the number of folds for cross-validation
    tscv = TimeSeriesSplit(n_splits=3)

    # iterating over folds, train model on each, forecast and calculate error
    for train, test in tscv.split(values):
        model = HoltWinters(
            series=values[train],
            slen=slen,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            n_preds=len(test),
        )
        model.triple_exponential_smoothing()

        predictions = model.result[-len(test):]
        actual = values[test]
        error = loss_function(predictions, actual)
        errors.append(error)

    return np.mean(np.array(errors))


def get_art():
    articles = 'datasets/custom_imports/editorial_planning.xlsx'
    art = pd.read_excel(articles, parse_dates=['Date'], index_col='Date')
    art.columns = ['title', 'theme']
    art = art.apply(lambda x: x.str.strip(' '))

    words = 'cap|challenge|quête|défi|challenge|concours'
    art.loc[(art.title.str.casefold().str.contains(words)), 'theme'] = 'Challenge_Quest_Related'

    custom_cat = ['Challenge_Quest_Related']
    cat_mask = ~(art.theme.isin(custom_cat))

    words = r'cooking|alimentation|bio|alimention|recette|cuisiner|nouvel an|menu|oeuf'

    food_mask = (art.theme.str.casefold().str.contains(words, regex=True) | art.title.str.casefold().str.contains(words,
                                                                                                                  regex=True))
    art.loc[food_mask & cat_mask, 'theme'] = 'Food'

    custom_cat.append('Food')
    cat_mask = ~(art.theme.isin(custom_cat))

    mask = (art.title.str.casefold().str.contains('journée|jour')) | (art.theme.str.contains('Nöel'))
    art.loc[mask & cat_mask, 'theme'] = 'Today_Event'

    custom_cat.append('Today_Event')
    cat_mask = ~(art.theme.isin(custom_cat))

    m = ((art.applymap(lambda x: 'tourisme' in x.casefold())).sum(axis=1) > 0)
    m2 = art.title.str.casefold().str.contains('mobil')
    art.loc[((m | m2) & cat_mask), 'theme'] = 'Mobility'

    custom_cat.append('Mobility')
    cat_mask = ~(art.theme.isin(custom_cat))

    mask = (art.title.str.contains(r'menu (de )?WAG', regex=True))
    art.loc[mask & cat_mask, 'theme'] = 'WAG_menu'

    custom_cat.append('WAG_menu')
    cat_mask = ~(art.theme.isin(custom_cat))

    themes = ['Consommation éco-responsable', 'Attitude éco-responsable',
              'Anti-gaspillage', 'Tri des déchets  / Recyclage']
    c1 = (art.theme.isin(themes)) | (art.title.str.casefold().str.contains('idées|guide|astuce|conseil|tip|coach'))
    c2 = ~((art.title.str.contains('#')) | (art.theme.isin(custom_cat)))
    mask = c1 & c2
    art.loc[mask, 'theme'] = 'Practical_tip'

    custom_cat.append('Practical_tip')
    cat_mask = ~(art.theme.isin(custom_cat))

    m1 = art.title.str.casefold().str.contains(r'bébés?|parents?|enfants?|école|rentrée', regex=True)
    m2 = art.theme.str.casefold().str.contains(r'(é|e)ducation', regex=True)
    art.loc[(m1 | m2) & cat_mask, 'theme'] = 'Education_family'

    custom_cat.append('Education_family')
    cat_mask = ~(art.theme.isin(custom_cat))

    mask = (art.theme.isin(['Protection des océeans', 'Protection de la nature',
                            'Environnement terrestre', 'Environnement maritime', 'Protection des animaux']))

    art.loc[mask & cat_mask, 'theme'] = 'Ecology_general'

    custom_cat.append('Ecology_general')
    cat_mask = ~(art.theme.isin(custom_cat))

    art.loc[cat_mask, 'theme'] = 'Miscellaneous'

    return art


def get_sub(serie, from_date):
    return serie.loc[from_date:]
