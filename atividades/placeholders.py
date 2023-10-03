import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# placeholders são espaços reservados para dados de entrada
# muito utilizados na versão 1.x. Na versão 2.x foram removidos.

p = tf.placeholder('float',None)
operacao = p + 2

with tf.Session() as sess:
    resultado = sess.run(operacao, feed_dict = {p:[1,2,3]})
    print(resultado)
    
p2  = tf.placeholder('float',[None,5]) #não informamos o numero de linhas
operacao2 = p2*5

with tf.Session() as sess:
    dados = [[1,2,3,4,5],[6,7,8,9,10]]
    resultado = sess.run(operacao2,feed_dict = {p2:dados})
    print(resultado)