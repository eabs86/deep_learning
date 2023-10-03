import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

entradas = tf.constant([-1.0,7.0,5.0], name = "entradas")
pesos = tf.constant([0.8,0.1,0.0], name = "pesos")

produto_escalar = tf.multiply(entradas,pesos)

somatorio = tf.reduce_sum(produto_escalar)

with tf.Session() as sess:
    print(sess.run(produto_escalar))
    print(sess.run(somatorio))
