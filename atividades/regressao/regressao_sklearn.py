import numpy as np
import matplotlib.pyplot as plt

X = np.array([[18],[23],[28],[33],[38],[43],[48],[53],[58],[63]]) #idades

y = np.array([[871],[1132],[1042],[1356],[1488],[1638],[1569],[1754],[1866],[1900]])

plt.scatter(X,y)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X,y)

#coeficientes: y = b0 + b1*X

#b0 - parâmetro que intercepta o eixo Y
regressor.intercept_

#b1 - declive, coeficiente angular

regressor.coef_


# previsão do valor do plano para um pessoa com idade de 21 anos
idade = np.array([40]).reshape(-1,1)
previsao = regressor.predict(idade)
print(previsao)

previsao2 = regressor.predict(X)
print(previsao2)

resultado = abs(y-previsao2)
print(resultado)

media = resultado.mean()

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y, previsao2) #métrica boa para visualização

mse = mean_squared_error(y, previsao2) #metrica boa para treinamento

plt.plot(X,y,'o')
plt.plot(X, previsao2,'-',color='red')
plt.title('Regressao Linear Simples')
plt.xlabel('Idade(anos)')
plt.ylabel('Valor do Plano (R$)')