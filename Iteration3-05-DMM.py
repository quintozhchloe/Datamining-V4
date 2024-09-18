"""
Created on Wed May  1 15:26:43 2024

@author: Chloe huang
"""


#5.1

#5.2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Transformed_dataset.csv')
data.drop(columns=['Human Development Groups'], inplace=True)


sns.pairplot(data[['Population Below National Poverty Line', 'Living Standards', 'Health Deprivation', 'Average_income_inequality','Gender Inequality Index (GII)']])
plt.show()


corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


X = data[['Living Standards', 'Health Deprivation', 'Average_income_inequality','Gender inequality index (Gll)']]
y = data['Population Below National Poverty Line']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and model evaluation
y_pred = model.predict(X_test)
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))


