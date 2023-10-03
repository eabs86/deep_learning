import tensorflow as tf

# Definir constantes
valor1 = tf.constant(2)
valor2 = tf.constant(3)

# Soma
soma = valor1 + valor2

# Resultado diretamente com eager execution
resultado = soma.numpy()
print(resultado)


# Definir variáveis
valor1 = tf.Variable(10, name='valor1')
valor2 = tf.Variable(3, name='valor2')

# Soma
soma = valor1 + valor2

# Executar o cálculo diretamente
resultado = soma.numpy()
print(resultado)