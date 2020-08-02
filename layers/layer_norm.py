import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, hidden_size, axis=[-2, -1], learnable=True):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size
        self.axis = axis
        self.learnable = learnable

    def build(self, input_shape):
        self.gamma = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.beta = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, epsilon=1e-6, input_dtype=tf.float32):
        mean = tf.reduce_mean(x, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=self.axis, keepdims=True)
        normalized = (x - mean) * tf.math.rsqrt(variance + epsilon)
        if self.learnable:
            return tf.cast(normalized * self.gamma + self.beta, input_dtype)

        if not self.learnable:
            return tf.cast(normalized, input_dtype)


class InstanceNormalization(LayerNormalization):
    def __init__(self, hidden_size, learnable=True):
        super(InstanceNormalization, self).__init__(hidden_size, axis=[-2], learnable=learnable)

    # def call(self, x, epsilon=1e-6, input_dtype=tf.float32):
    #     print('norm', tf.shape(x))
    #     return super(InstanceNormalization, self).call(x, epsilon=1e-6, input_dtype=tf.float32)
