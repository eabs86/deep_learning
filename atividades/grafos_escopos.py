import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()

#otimo para problemas complexos.
with tf.name_scope('Operacoes'):
    with tf.name_scope("Escopo_A"):
        a = tf.add(2,2, name = 'add')
    with tf.name_scope("Escopo_B"):
        b = tf.multiply(a,3,name="mult1")
        c = tf.multiply(b,a,name = 'mult2')



with tf.Session() as sess:
    writer = tf.summary.FileWriter('output_scope', sess.graph)
    print(sess.run(c))
    writer.close()

#para criar novos grafos: tf.Graph()
# o grafo default está em tf.get_default_graph
# Isso é muito utilizado para paralelimos, onde os grafos podem ser executados
# de forma paralela.

# para definir outro grafo como padrao: grafo2.as_default()