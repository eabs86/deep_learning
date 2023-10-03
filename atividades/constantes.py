# usando o tensorflow 2.x


import tensorflow as tf
tf.__version__

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

a = tf.Variable(0.41)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(a.eval())
    

p = tf.placeholder(tf.float32,[1,1])
p


# trabalhando com constantes

valor1= tf.constant(2)
valor2 = tf.constant(3)

type(valor1)
print(valor1)


soma = valor1 + valor2

print(soma)

#necessário rodar os dados dentro de uma sessão
with tf.Session() as sess:
    s = sess.run(soma)

print(s)

texto1 = tf.constant('Texto 1')

texto2 = tf.constant('Texto 2')

print(texto1)

with tf.Session() as sess:
    con =sess.run(texto1 + texto2)
    
print(con)
