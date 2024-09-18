"""
# -*- coding: utf-8 -*-
Created on Wed May  1 15:26:43 2024

@author: Chloe huang
"""


#4.1 Data Reduction 
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split

cd = pd.read_csv('Clean_dataset_new.csv')


X = cd.drop('Population Below National Poverty Line', axis=1)  # Predictor features
y = cd['Population Below National Poverty Line']  # Target variable

X = pd.get_dummies(X)

selector = SelectKBest(score_func=f_regression, k=10)  
X_selected = selector.fit_transform(X, y)

selected_features_df = pd.DataFrame(selector.inverse_transform(X_selected), 
                                     columns=X.columns, 
                                     index=cd.index)
selected_columns = selected_features_df.columns[selected_features_df.var() != 0]
print("Selected Features:", selected_columns)

selected_columns = selected_columns.drop(['Human Development Groups_High', 'Human Development Groups_Low'])
#reduce unuseful data
reduced_df = cd[selected_columns.tolist() + ['Population Below National Poverty Line']+['Human Development Groups']]
reduced_df.to_csv('Reduced_data.csv', index=False)

reduced_df.shape
reduced_df.info()
reduced_df.head()
reduced_df.dtypes

#4.2 Data Transformation

import pandas as pd
import numpy as np

# Load the reduced dataset
df = pd.read_csv('Reduced_data.csv')

# Applying logarithmic transformation to the target variable
# We add 1 to avoid taking log of zero
df['Log_Population Below National Poverty Line'] = np.log(df['Population Below National Poverty Line'] + 1)

# Optionally save the transformed dataset
df.to_csv('Transformed_dataset.csv', index=False)

print(df[['Population Below National Poverty Line', 'Log_Population Below National Poverty Line']].head())


