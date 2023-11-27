import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn import metrics

df = pd.read_csv("/content/weatherHistory.csv")
df = df.drop_duplicates(['Formatted Date'], keep='first')
df.sort_values(by=['Formatted Date'], inplace=True)
df.reset_index(inplace=True, drop=True)
df = df.drop(columns=['Formatted Date', 'Loud Cover'])

df = df[df['Precip Type'].notna()]
df = df[df['Humidity'] != 0.0]
df = df[df['Pressure (millibars)'] != 0]
df.reset_index(inplace=True, drop=True)

corr_matrix = df.corr()
df["temp_cat"] = pd.cut(df["Apparent Temperature (C)"], bins=[-np.inf, 0, 10, 20, np.inf], labels=[1, 2, 3, 4])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["temp_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

strat_train_set.drop("temp_cat", axis=1, inplace=True)

df_train = strat_train_set.drop("Apparent Temperature (C)", axis=1)
df_train_labels = strat_train_set["Apparent Temperature (C)"].copy()

num_attribs = list(df_train.select_dtypes(include=[np.number]))
cat_attribs = ["Precip Type", "Summary", "Daily Summary"]

num_pipeline = Pipeline([('std_scaler', StandardScaler())])
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs), ("cat", OrdinalEncoder(), cat_attribs)])
df_train_prepared = full_pipeline.fit_transform(df_train)

lin_reg = LinearRegression()
lin_reg.fit(df_train_prepared, df_train_labels)
df_train_predictions = lin_reg.predict(df_train_prepared)
accuracy_score = lin_reg.score(df_train_prepared, df_train_labels)

poly_reg = PolynomialFeatures(degree=5)
poly_transform = poly_reg.fit_transform(df_train_prepared)
lin_reg2 = LinearRegression()
lin_reg2.fit(poly_transform, df_train_labels)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(df_train_prepared, df_train_labels)

scores = cross_val_score(tree_reg, df_train_prepared, df_train_labels, scoring="neg_mean_squared_error", cv=10)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(df_train_prepared, df_train_labels)

forest_scores = cross_val_score(forest_reg, df_train_prepared, df_train_labels, scoring="neg_mean_squared_error", cv=10)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(df_train_prepared, df_train_labels)

param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8)}
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(df_train_prepared, df_train_labels)

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("Apparent Temperature (C)", axis=1)
y_test = strat_test_set["Apparent Temperature (C)"].copy()
X_test_prepared = full_pipeline.fit_transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
accuracy = metrics.r2_score(y_test, final_predictions)
