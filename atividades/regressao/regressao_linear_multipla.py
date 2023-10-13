import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

dataset = pd.read_csv('house_prices.csv')

#queremos fazer a previsao do preço da casa pela regressao linear multipla

#y = b0 + b1*x1 + b2*x2 + ... + bn*xn

colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

dataset = dataset.loc[:,colunas_usadas]

X = dataset.iloc[:,1:]
y = dataset.iloc[:,0:1]

from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
colunas_x = ['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
X = pd.DataFrame(X,columns=colunas_x)

y['price'] = pd.DataFrame(scaler_y.fit_transform(y))

previsores_colunas = colunas_x

colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]
#basicamente itera as colunas e vai criando cada elemento no tensor flow.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = X_train,
                                                          y=y_train,
                                                          batch_size = 32,
                                                          num_epochs = None,
                                                          shuffle = True)

funcao_teste = tf.estimator.inputs.pandas_input_fn(x = X_test,
                                                          y=y_test,
                                                          batch_size = 32,
                                                          num_epochs = 10000,
                                                          shuffle = False)

regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

regressor.train(input_fn = funcao_treinamento,steps = 10000)
