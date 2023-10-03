import tensorflow as tf

# Definir vetor
vetor = tf.constant([1,2,3],name='vetor')


# Somar
soma = tf.Variable(vetor + 5, name = 'soma')

# Resultado diretamente com eager execution
resultado = soma.numpy()
print(resultado)

valor = tf.Variable(0,name='valor')

for i in range(5):
    valor = valor + 1
    print(int(valor))

