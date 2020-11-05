import tensorflow as tf    

class Dense(tf.Module):
    def __init__(
        self,
        in_features,
        out_features,
        name = None,
        initializer = tf.initializers.GlorotNormal(),
        activation = tf.nn.relu,
    ):
        super().__init__(name=name)

        self.activation = activation

        self.w = tf.Variable(
            initializer(in_features, out_features),
            dtype = tf.float32,
            name = 'w',
        )

        self.b = tf.Variable(
            tf.zeros([out_features], name='b')
        )
        
    def __call__(self, x, training=None):
        y = tf.matmul(x, self.w) + self.b
        return self.activation(y)

class InvDropout(tf.Module):
    """
    Inverted Dropout is Dropout where the activations are rescaled in order to keep
        the maginitude of the activations similar in both training and 
    """
    def __init__(
        self,
        p,
        name=None,
    ):
        super().__init__(name=name)
        self.p = p
        self.scale = 1/(1-p)

    def __call__(self, x, training=None):
        if training:
            mask = tf.random.uniform(
                shape = x.shape[1:],
                minval = 0.0,
                maxval = 1.0,
            ) < self.p
            x = self.scale * X * mask
        return x

    