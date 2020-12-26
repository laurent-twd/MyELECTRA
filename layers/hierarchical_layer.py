import tensorflow as tf 

class HierarchicalLayer(tf.keras.layers.Layer):

    def __init__(self, vocab_size, enc_units):

        super(HierarchicalLayer, self).__init__()
        self.input_dim = enc_units
        self.output_dim = vocab_size

        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.output_dim, self.input_dim), dtype="float32"),
            trainable=True,
        )

    def call(self, x, z, h):
        x = tf.cast(x, dtype = tf.int32)
        y = tf.sigmoid(tf.reduce_sum(tf.gather(self.w, x) * tf.expand_dims(h, axis = 2), axis = 3) * tf.cast(z, dtype = tf.float32))
        mask = tf.not_equal(x, 0)
        y = tf.where(mask, y, 1.)
        return tf.reduce_prod(y, axis = 2)
