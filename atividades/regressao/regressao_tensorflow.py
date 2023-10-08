import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]]) #idades

y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]]) #valor do plano

#Escalonamento das variáveis (necessário para o tensorflow)

from sklearn.preprocessing import StandardScaler

scaler_x = StandardScaler()

X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()

y = scaler_y.fit_transform(y)

plt.scatter(X,y)

# Criando a fórmula para regressao linear simples

# y = b0 + b1 * X


b0 = tf.Variable(0.5488, name='b0')
b1 = tf.Variable(0.7151,name='b1')

erro  = tf.losses.mean_squared_error(y, (b0+b1*X))

otimizador = tf.train.GradientDescentOptimizer(learning_rate = 0.001)

treinamento = otimizador.minimize(erro)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(treinamento)
    
    b0_final, b1_final = sess.run([b0,b1])
    
previsoes=b0_final + b1_final*X

plt.plot(X,y,'o')
plt.plot(X,previsoes,'-',color='red')

#para realizar previsão em novos inputs, é necesário escalonar!

idade = scaler_x.transform([[32]])

previsao = b0_final + b1_final*idade

previsao_inverse_scaled = scaler_y.inverse_transform(previsao)

y_inversed_scaled = scaler_y.inverse_transform(y)

previsoes_inversed_scaled = scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_inversed_scaled,previsoes_inversed_scaled) 

mse = mean_squared_error(y_inversed_scaled,previsoes_inversed_scaled) 
