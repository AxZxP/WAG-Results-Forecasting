from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


def feature_extraction(filepath=None):
    """
    input: a datefile path
    output: a cleaned dataframe
    """
    df = pd.read_csv(filepath, parse_dates=True, index_col=['Start Date'],
                     usecols=['Start Date', 'Users', 'Week_0', 'Week_1', 'Week_2', 'Week_3', 'Week_4', 'Week_8'])
    df.rename({'Week_8': 'y', 'Users': 'Cohort_size'}, axis=1, inplace=True)
    df['Month'] = df.index.month_name()
    df = df.loc[~((df == 0).any(axis=1))]
    df['Cohort_active_users'] = df.Week_0 / df.Cohort_size * 100
    sub = df[['Week_0', 'Week_1', 'Week_2', 'Week_3', 'Week_4', 'y']].apply(lambda x: x / df.Week_0 * 100)
    df.loc[:, ['Week_0', 'Week_1', 'Week_2', 'Week_3', 'Week_4', 'y']] = sub
    df.drop('Week_0', axis=1, inplace=True)
    df.Cohort_size = df.Cohort_size.map(np.log)

    return df


def get_the_set(df, target_variable):
    """
    input: a matrix with all variables and a label for the y variable
    output: X_train, X_test, y_train, y_test
    """
    X = df.drop(labels=target_variable, axis=1).copy()
    y = df[f'{target_variable}'].copy()

    X_train, X_test, y_train, y_test = map(lambda x: x.reset_index(drop=True), train_test_split(X, y,
                                                                                                test_size=1 / 4,
                                                                                                random_state=42,
                                                                                                # stratify=y
                                                                                                ))

    return X_train, X_test, y_train, y_test


def fit_encoder(df):
    """
    input: a matrix with all variables
    output: a fitted encoder
    """
    encoder = OneHotEncoder(drop='first')  # , handle_unknown='ignore')
    cat_vars = df.select_dtypes(include=['O', 'category'])
    encoder.fit(cat_vars)
    return encoder


def run_scaling(X):
    """
    input: a matrix with all variables
    output: a matrix with all numerical values scaled
    """
    scaler = StandardScaler()
    num_var = X.select_dtypes(include=['int', 'float']).copy()
    num_scaled = scaler.fit_transform(num_var)
    X.loc[:, X.select_dtypes(include=['int', 'float']).columns] = num_scaled
    return X


def encode_categorical_variables(X, encoder):
    """
    input: a matrix with all variables
    output: a matrix with categorical and object encoded with (0,1)
    """
    cat_vars = X.select_dtypes(include=['O', 'category']).copy()
    cat_names = cat_vars.columns
    try:
        encoded = encoder.transform(cat_vars).toarray()
    except ValueError as e:
        return X
    df_cat = pd.DataFrame(encoded, columns=encoder.get_feature_names(cat_names))
    X = X.select_dtypes(exclude=['O', 'category']).join(df_cat)
    return X


# Data prep

data_file = '.../cohort_analysis.csv'
df = feature_extraction(data_file).drop('Month', axis=1)
X_train, X_test, y_train, y_test = get_the_set(df=df, target_variable='y')

encoder = fit_encoder(X_train)

X_train = run_scaling(X_train)
X_train = encode_categorical_variables(X_train, encoder)

X_test = run_scaling(X_test)
X_test = encode_categorical_variables(X_test, encoder)

# Model
model = OLS(y_train, add_constant(X_train.drop(['Week_1', 'Week_3', 'Cohort_active_users'], axis=1))).fit()
model.summary()

# Test the assumption of Linear Regression
tester = Assumptions.Assumption_Tester_OLS(X_train, y_train)
tester.run_all()
