import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


#vetores
a = tf.constant([1,2,3],name = 'vetor_a')
b = tf.constant([4,5,6],name = 'vetor_b')

soma_vetor = tf.Variable(a + b, name = 'soma_vetores')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(soma_vetor))
    
    
#matrizes

matriz_a = tf.constant([[1,2,3,4],[5,6,7,8],[9,10,11,12]], name = "matriz_a")
matriz_b = tf.constant([[1,1,1,1],[2,2,2,2],[3,3,3,3]], name = "matriz_b")

soma_matriz = matriz_a + matriz_b

with tf.Session() as sess:
    print(sess.run(matriz_a))
    print(sess.run(matriz_b))
    print(sess.run(soma_matriz))

