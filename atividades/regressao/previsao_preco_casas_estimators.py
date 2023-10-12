import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

dataset = pd.read_csv('house_prices.csv')

#queremos fazer a previsao do preço da casa

dataset.count()

X = dataset.iloc[:,5].values

X = X.reshape(-1,1)

y = dataset.iloc[:,2:3]

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

colunas = [tf.feature_column.numeric_column('x', shape=[1])]

regressor = tf.estimator.LinearRegressor(feature_columns = colunas)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

funcao_treinamento = tf.estimator.inputs.numpy_input_fn({'x':X_train},
                                                        y_train,
                                                        batch_size=32,
                                                        num_epochs = None,
                                                        shuffle = True)

funcao_testes = tf.estimator.inputs.numpy_input_fn({'x':X_test},
                                                    y_test,
                                                    batch_size=32,
                                                    num_epochs = 1000,
                                                    shuffle = False)

regressor.train(input_fn = funcao_treinamento,steps = 10000)

metricas_treinamento = regressor.evaluate(input_fn = funcao_treinamento,
                                          steps = 10000)

metricas_teste = regressor.evaluate(input_fn = funcao_testes,
                                    steps = 10000)

metricas_treinamento
metricas_teste

novas_casas = np.array([[800], [900], [1000]]).reshape(-1,1)
novas_casas

novas_casas = scaler_x.transform(novas_casas)
novas_casas

funcao_previsao = tf.estimator.inputs.numpy_input_fn({'x':novas_casas}, shuffle = False)
previsoes = regressor.predict(input_fn = funcao_previsao)

for p in regressor.predict(input_fn = funcao_previsao):
    print(scaler_y.inverse_transform(p['predictions'].reshape(-1,1)))