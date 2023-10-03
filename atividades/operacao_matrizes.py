import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#operação entre matrizes

matriz_a = tf.constant([[1,2,3],[5,6,7],[9,10,11]], name = "matriz_a")
matriz_b = tf.constant([[1,1,1],[2,2,2],[3,3,3]], name = "matriz_b")

multiplicacao = tf.matmul(matriz_a,matriz_b)

#se usar matriz_a * matriz_b é outro tipo de operação, e não multiplicação de matriz
operacao_2 = matriz_a * matriz_b

with tf.Session() as sess:
    print(sess.run(multiplicacao))
    print('\n')
    print(sess.run(operacao_2))

