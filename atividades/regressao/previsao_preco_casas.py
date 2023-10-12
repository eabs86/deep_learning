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

plt.scatter(X,y)

#Regressão Linear Simples
# y = b0 + b1*x
np.random.seed(1)
print(np.random.rand(2))

b0 = tf.Variable(0.4170, name='b0')
b1 = tf.Variable(0.7203, name='b1')

#usando placeholders
batch_size = 32
xph = tf.placeholder(tf.float32, [batch_size,1])
yph = tf.placeholder(tf.float32,[batch_size,1])


y_modelo = b0 + b1*xph
erro = tf.losses.mean_squared_error(yph,y_modelo)
otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        indices = np.random.randint(len(X), size = batch_size)
        feed = {xph: X[indices], yph:y[indices]}
        sess.run(treinamento,feed_dict = feed)
        b0_final, b1_final = sess.run([b0,b1])

previsoes = b0_final + b1_final * X

plt.plot(X,y,'o')
plt.plot(X,previsoes, color='red')

y1 = scaler_y.inverse_transform(y)

previsoes1 = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, previsoes1)
