# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from statsmodels.api import OLS, add_constant
# %%
boston = load_boston()
boston
# %%
X = boston['data']
y = boston['target']
X_copy = pd.DataFrame(X, columns=boston['feature_names'])
y_copy = pd.Series(y, name="MEDV")
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# %%
model_1 = LinearRegression()
model_1.fit(Xtrain, ytrain)
ytest_pred = model_1.predict(Xtest)
mse = mean_squared_error(ytest, ytest_pred)
print(mse)
# %%
Xtrain_new = add_constant(Xtrain)
Xtest_new = add_constant(Xtest)
Xtrain_new_copy = pd.DataFrame(Xtrain_new)
Xtrain_new_copy.head()
# %%
model_2 = OLS(ytrain, Xtrain_new)
result = model_2.fit()
ytest_pred_2 = result.predict(Xtest_new)
mse_2 = mean_squared_error(ytest, ytest_pred_2)
print(mse, mse_2)
# %%
print(result.summary())
