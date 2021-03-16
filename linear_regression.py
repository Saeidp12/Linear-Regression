# %%
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# %%
dataframe = pd.read_csv("E:/dataframe1.csv")
dataframe.head(6)
# %%
X_cols = ['X1', 'X2']
X = dataframe[X_cols]
y = dataframe['y']

# %%
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=1234)

# %%
model = LinearRegression()
model.fit(Xtrain, ytrain)
ytest_pred = model.predict(Xtest)

# %%
mse = mean_squared_error(ytest, ytest_pred)
print(f"Mean Squared Error: {mse}")
# %%
