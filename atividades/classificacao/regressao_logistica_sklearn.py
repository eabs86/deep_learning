import pandas as pd


base = pd.read_csv('census.csv')

base['income'].unique()

X = base.iloc[:,0:14].values

y = base.iloc[:,14].values

# Pre-processamento dos dados categóricos

from sklearn.preprocessing import LabelEncoder


labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])
X[:, 7] = labelencoder_X.fit_transform(X[:, 7])
X[:, 8] = labelencoder_X.fit_transform(X[:, 8])
X[:, 9] = labelencoder_X.fit_transform(X[:, 9])
X[:, 13] = labelencoder_X.fit_transform(X[:, 13])

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()

X = scaler_X.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import LogisticRegression

classificador = LogisticRegression(max_iter = 1000)

classificador.fit(X_train, y_train)

previsoes = classificador.predict(X_test)

from sklearn.metrics import accuracy_score

taxa_acerto = accuracy_score(y_test, previsoes)