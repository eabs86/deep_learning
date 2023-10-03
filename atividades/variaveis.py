import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


valor1 = tf.constant(15, name = 'valor1')

print(valor1)

#inicialização de variavel


soma = tf.Variable(valor1 + 5, name = 'result_soma')

init = tf.global_variables_initializer()
print(soma)

with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma)

print(s)
