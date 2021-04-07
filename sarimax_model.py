from helpers.process import *

# Import the data
art = get_art()
df, act, quizz = get_dfs()
art_encode = pd.get_dummies(art.theme).resample('D').sum()

data = act.merge(art_encode, how='left',
                 left_on=act.index,
                 right_on=art_encode.index).fillna(0).set_index('key_0')
data.index.name = 'DateTime'
y = data.Actions
X = data.drop('Actions', axis=1)

# Test/Train Split

X_train, X_test, y_train, y_test = X.iloc[:-30], X.iloc[-30:], y.iloc[:-30], y.iloc[-30:]
ex0 = X_train

# setting initial values and some bounds for them
d = 1
s = 7

m = (sm.tsa.statespace.SARIMAX(y_train, order=(0, d, 2),
                               seasonal_order=(1, 0, 1, 7),
                               exog=ex0).fit(disp=-1))

print(m.summary())
