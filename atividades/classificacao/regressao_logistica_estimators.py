import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.model_selection import train_test_split

base = pd.read_csv('census.csv')

base['income'].unique()

def converte_classe(rotulo):
    if rotulo == '>50k':
        return 1
    else:
        return 0
    
base['income']= base['income'].apply(converte_classe)

X = base.drop('income', axis = 1)

y = base['income']

#definindo faixas de idades

base.age.hist()


idade = tf.feature_column.numeric_column('age')

idade_categorica = [tf.feature_column.bucketized_column(
    idade, boundaries = [20, 30, 40, 50, 60, 70, 80, 90])]

X.columns

nome_colunas_categoricas = ['workclass',
                       'education',
                       'marital-status',
                       'occupation',
                       'relationship',
                       'race',
                       'sex',
                       'native-country']

colunas_categoricas = [tf.feature_column.categorical_column_with_vocabulary_list(
    key=coluna, vocabulary_list= X[coluna].unique()) for coluna in nome_colunas_categoricas]


nomes_colunas_numericas = ['age', 'capital-gain', 'capital-loos', 'hour-per-week']
colunas_numericas = [tf.feature_column.numeric_column(
    key = coluna) for coluna in nomes_colunas_numericas]

colunas = idade_categorica + colunas_categoricas + colunas_numericas

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


funcao_treinamento = tf.estimator.inputs.pandas_input_fn(
    x = X_train,
    y = y_train,
    batch_size = 32,
    num_epochs = None,
    shuffle = True
)   

classificador = tf.estimator.LinearClassifier(feature_columns = colunas)
classificador.train (input_fn = funcao_treinamento, steps = 10000)

funcao_previsao = tf.estimator.inputs.pandas_input_fn(
    x = X_test,
    y = y_test,
    batch_size = 32,
    shuffle = False
)

previsoes = classificador.predict(input_fn = funcao_previsao)

list(previsoes)

previsoes_final = []

for p in classificador.predict(input_fn = funcao_previsao):
    previsoes_final.append(p['class_ids'])

previsoes_final

from sklearn.metrics import accuracy_score

accuracy_score(y_test, previsoes_final)