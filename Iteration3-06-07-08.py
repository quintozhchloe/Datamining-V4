# -*- coding: utf-8 -*-
"""
Created on Fri May  2 01:18:06 2024

@author: Chloe huang
"""



#6.1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


data = pd.read_csv('Transformed_dataset.csv')


X = data.drop(['Population Below National Poverty Line'], axis=1)
y = data['Population Below National Poverty Line']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# SVR
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)

# MLPRegressor
nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)
nn_model.fit(X_train, y_train)
y_pred_nn = nn_model.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred_nn)

# output
print(f'Linear Regression RMSE: {np.sqrt(mse_lr)}')
print(f'SVR RMSE: {np.sqrt(mse_svm)}')
print(f'Neural Network RMSE: {np.sqrt(mse_nn)}')


#6.2
#6.3
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#6.3.1
data.drop(columns=['GII Rank', 'Population Below $1.25 per Day'], inplace=True)
#data.to_csv('Transformed_dataset.csv', index=False)

data = pd.read_csv('Transformed_dataset.csv')
X = data.drop(['Population Below National Poverty Line'], axis=1)
y = data['Population Below National Poverty Line']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use Ridge
poly_ridge_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
poly_ridge_model.fit(X_train, y_train)
y_pred_poly_ridge = poly_ridge_model.predict(X_test)
mse_poly_ridge = mean_squared_error(y_test, y_pred_poly_ridge)
print(f'Polynomial Ridge RMSE: {np.sqrt(mse_poly_ridge)}')

# use Lasso
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f'Lasso RMSE: {np.sqrt(mse_lasso)}')


#6.3.3-model
from sklearn.model_selection import GridSearchCV

# Param
param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],  # try 1,2,3 degress
    'ridge__alpha': [0.1, 1.0, 10.0]  # try alpha
}

# GridSearchCV
grid_search = GridSearchCV(poly_ridge_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

#
best_model = grid_search.best_estimator_
best_score = np.sqrt(-grid_search.best_score_)
print(f'Best model parameters: {grid_search.best_params_}')
print(f'Best model RMSE: {best_score}')


#6.3.4
from sklearn.model_selection import GridSearchCV


param_grid = {
    'polynomialfeatures__degree': [2],
    'ridge__alpha': np.logspace(-2, 1, 20)  # 从0.01到10，共20个值
}


grid_search = GridSearchCV(poly_ridge_model, param_grid, cv=10, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_
best_score = np.sqrt(-grid_search.best_score_)
print(f'Best model parameters: {grid_search.best_params_}')
print(f'Best model RMSE: {best_score}')


y_pred_final = best_model.predict(X_test)
final_rmse = mean_squared_error(y_test, y_pred_final, squared=False)
print(f'Final test RMSE: {final_rmse}')




#7.1
 
#Create Logical Test Designs

import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.01438449888287663))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

rmse_scores = np.sqrt(-scores)
print(f'Cross-validated RMSE: {np.mean(rmse_scores)} +/- {np.std(rmse_scores)}')



# 7.2: Conduct Data Mining

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


best_model.fit(X_train, y_train)


y_pred = best_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE: {rmse}')


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted', linestyle='--')
plt.title('Comparison of Actual and Predicted Values')
plt.legend()
plt.show()


#7.3 patterns

#1.
import numpy as np
import matplotlib.pyplot as plt


errors = y_test - y_pred


plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

print("Error Statistics:")
print(f"Mean Error: {np.mean(errors)}")
print(f"Median Error: {np.median(errors)}")
print(f"Standard Deviation of Errors: {np.std(errors)}")


#2.

import seaborn as sns
X_test.info()
plt.figure(figsize=(10, 6))
sns.boxplot(x=X_test['Living Standards'], y=errors)
plt.title('Error Distribution across Categorical Feature1')
plt.xlabel('Category')
plt.ylabel('Prediction Error')
plt.show()

sns.boxplot(x=X_test['Gender Inequality Index (GII)'], y=errors)
plt.title('Error Distribution across Categorical Feature2')
plt.xlabel('Category')
plt.ylabel('Prediction Error')
plt.show()

sns.boxplot(x=X_test['Population with Secondary Education (Female)'], y=errors)
plt.title('Error Distribution across Categorical Feature3')
plt.xlabel('Category')
plt.ylabel('Prediction Error')
plt.show()

sns.boxplot(x=X_test['Health Deprivation'], y=errors)
plt.title('Error Distribution across Categorical Feature4')
plt.xlabel('Category')
plt.ylabel('Prediction Error')
plt.show()

sns.boxplot(x=X_test['Average_income_inequality'], y=errors)
plt.title('Error Distribution across Categorical Feature5')
plt.xlabel('Category')
plt.ylabel('Prediction Error')
plt.show()
#3

with open('model_performance_report.txt', 'w') as file:
    file.write("Model Performance Report\n")
    file.write("-------------------------------\n")
    file.write(f"Mean Error: {np.mean(errors)}\n")
    file.write(f"Median Error: {np.median(errors)}\n")
    file.write(f"Standard Deviation of Errors: {np.std(errors)}\n")


plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.savefig('error_distribution.png')




#8.1
#8.1.1
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(12, 6))
sns.histplot(errors, kde=True)
plt.title('Overall Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.show()


for feature in ['Living Standards', 'Gender Inequality Index (GII)', 
                'Population with Secondary Education (Female)', 'Health Deprivation', 
                'Average_income_inequality']:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=X_test[feature], y=errors)
    plt.title(f'Error Distribution across {feature}')
    plt.xlabel(feature)
    plt.ylabel('Prediction Error')
    plt.show()


#8.1.1

import numpy as np

print("Detailed Analysis of Model Performance by Features:")
for feature in ['Living Standards', 'Gender Inequality Index (GII)', 
                'Population with Secondary Education (Female)', 'Health Deprivation', 
                'Average_income_inequality']:
    mean_error = np.mean(errors[X_test[feature] == X_test[feature].median()])
    std_error = np.std(errors[X_test[feature] == X_test[feature].median()])
    print(f"{feature} - Mean Error at Median Category: {mean_error}, Std Dev: {std_error}")
    
if errors[X_test[feature] == X_test[feature].median()].size > 0:
    mean_error = np.mean(errors[X_test[feature] == X_test[feature].median()])
    std_error = np.std(errors[X_test[feature] == X_test[feature].median()])
    print(f"{feature} - Mean Error at Median Category: {mean_error}, Std Dev: {std_error}")
else:
    print(f"{feature} - No data available for this category.")

#8.2
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of key features
plt.figure(figsize=(12, 6))
sns.histplot(X_train['Living Standards'], kde=True, color='blue')
plt.title('Distribution of Living Standards')
plt.xlabel('Living Standards')
plt.ylabel('Frequency')
plt.show()

# Visualize prediction errors
plt.figure(figsize=(12, 6))
sns.histplot(errors, kde=True, color='red')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Density')
plt.show()

# Relationship between features and prediction error
features = ['Gender Inequality Index (GII)', 'Health Deprivation', 'Average_income_inequality']
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=X_test[feature], y=errors)
    plt.title(f'Error Distribution across {feature}')
    plt.xlabel(feature)
    plt.ylabel('Prediction Error')
    plt.show()



#8.4
from sklearn.metrics import mean_squared_error, r2_score


rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Evaluation Metrics:")
print(f"Root Mean Squared Error: {rmse}")
print(f"R^2 Score: {r2}")


from sklearn.model_selection import cross_val_score
cv_rmse = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
print(f"Cross-Validated RMSE: {np.mean(-cv_rmse)}")


#8.5

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline


data = pd.read_csv('Transformed_dataset.csv')
X = data.drop(['Population Below National Poverty Line'], axis=1)
y = data['Population Below National Poverty Line']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial Best Model
best_model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=0.01438449888287663))
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
initial_rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Initial RMSE: {initial_rmse}")

# Iteration 1: Adjusting the degree of polynomial features
for degree in [1, 2, 3]:
    model = make_pipeline(PolynomialFeatures(degree=degree), Ridge(alpha=0.01438449888287663))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE with Polynomial Degree {degree}: {rmse}")

# Iteration 2: Different alphas
alphas = np.logspace(-4, -1, 10)
for alpha in alphas:
    model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=alpha))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"RMSE with Alpha {alpha}: {rmse}")

# Iteration 3: Using GridSearchCV to adjust the model
param_grid = {
    'polynomialfeatures__degree': [1, 2, 3],
    'ridge__alpha': np.logspace(-4, -1, 10)
}
grid_search = GridSearchCV(make_pipeline(PolynomialFeatures(), Ridge()), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f"Best parameters from Grid Search: {grid_search.best_params_}")
best_rmse = mean_squared_error(y_test, grid_search.best_estimator_.predict(X_test), squared=False)
print(f"Best RMSE from Grid Search: {best_rmse}")





