
import tensorflow as tf
from forkED.layers import InvDropout, Dense

class MLPBottleneck(tf.Module):
    def __init__(
        self,
        in_dim,
        first_hidden_dim,
        emb_dim,
        n_h_layers,
        hidden_act=tf.nn.relu,
        bottleneck_act=tf.nn.relu,
        dropout=0.0,
        name=None
    ):
        super().__init__(name=name)
        dims = [in_dim] + [first_hidden_dim // (2 ** i) for i in range(n_h_layers - 1)]
        assert dims[-1] > emb_dim, (
            "the embedding dimension must be smaller than the dimension of the last hidden layer"
        )
        if isinstance(dropout,float):
            dropout = [dropout] * n_h_layers
        assert len(dropout) == n_h_layers, "the length of `dropout` must be the same as `n_h_layers`"
        self.layers = []
        for i, dim in enumerate(dims[:-1]):
            next_dim = dims[i + 1]
            layer = Dense(dim, next_dim, activation=hidden_act)
            self.layers.append(layer)
            if dropout[i] > 0.0 and dropout[i] < 1.0:
                droplayer = InvDropout(dropout[i])
                self.layers.append(droplayer)

        self.bottleneck = Dense(dims[-1], emb_dim, activation=bottleneck_act)

    def __call__(self, x, training=None):
        for layer in self.layers:
            x = layer(x, training=training)
        return self.bottleneck(x)

class MLPReverseBottleneck(tf.Module):
    def __init__(
        self,
        emb_dim,
        out_dim,
        n_h_layers,
        hidden_act=tf.nn.relu,
        out_act=tf.nn.relu,
        dropout=0.0,
        name=None
    ):
        super().__init__(name=name)
        dims = [emb_dim * (2 ** i) for i in range(n_h_layers)]
        assert dims[-1] < out_dim, (
            "the out dimension must be larger than the dimension of the last hidden layer"
        )
        if isinstance(dropout,float):
            dropout = [dropout] * n_h_layers
        assert len(dropout) == n_h_layers, "the length of `dropout` must be the same as `n_h_layers`"
        self.layers = []
        for i, dim in enumerate(dims[:-1]):
            next_dim = dims[i + 1]
            layer = Dense(dim, next_dim, activation=hidden_act)
            self.layers.append(layer)
            if dropout[i] > 0.0 and dropout[i] < 1.0:
                droplayer = InvDropout(dropout[i])
                self.layers.append(droplayer)

        self.output = Dense(dims[-1], out_dim, activation=out_act)

    def __call__(self, x, training=None):
        for layer in self.layers:
            x = layer(x, training=training)
        return self.output(x)


class EncoderDecoder(tf.Module):
    """
    in_dim:      dimension of the input data
    first_hidden_dim:   dimension of the first hidden layer of the encoder
    emb_dim:         dimension of the bottleneck of the encoder
    """
    def __init__(
        self,
        in_dim,
        first_hidden_dim,
        emb_dim,
        n_h_layers,
        dropout = 0.0,
        out_act = tf.nn.sigmoid,
        bottleneck_act = tf.nn.relu,
        name = None,
    ):
        super().__init__(name=name)
        self.encoder = MLPBottleneck(
            in_dim,
            first_hidden_dim,
            emb_dim,
            n_h_layers,
            dropout=dropout,
            bottleneck_act=bottleneck_act,
            name='encoder'
        )
        self.decoder = MLPReverseBottleneck(
            emb_dim,
            in_dim,
            n_h_layers,
            out_act=out_act,
            dropout=dropout,
            name='decoder'
        )

    def __call__(self, x, training=None):
        latent = self.encoder(x, training=training)
        return self.decoder(latent, training=training)

class ForkED(EncoderDecoder):
    """
    in_dim:      dimension of the input data
    first_hidden_dim:   dimension of the first hidden layer of the encoder
    emb_dim:         dimension of the bottleneck of the encoder
    fork_task_dim:      dimension of the output of the fork task
    dlow_dim:           dimension of the output of the lower dim construct
                            (high_dim construct always = input_dim)
    fork_task_act:      activation for output of fork task
    dhigh_act:          activation for output of high dim task 
    dlow_act:           activation for output of low dim task
    """
    def __init__(
        self,
        in_dim,
        first_hidden_dim,
        emb_dim,
        n_h_layers,
        fork_task_dim,
        dlow_dim,
        dropout=0.0,
        fork_task_act=tf.nn.sigmoid,
        dhigh_act=tf.nn.sigmoid,
        dlow_act=tf.nn.sigmoid,
        name=None,
    ):
        super().__init__(
            in_dim,
            first_hidden_dim,
            emb_dim,
            n_h_layers,
            dropout = droput,
            out_act = dhigh_act,
            name=name,
        )
        self.decoder_low = MLPReverseBottleneck(
            emb_dim,
            dlow_dim,
            n_h_layers,
            out_act=dlow_act,
            name='decoder_low'
        )
        self.fork = MLPBottleneck(
            emb_dim * 2,
            emb_dim,
            fork_task_dim,
            n_h_layers,
            out_act=fork_task_act,
            name='fork'
        )

    def get_embedding(self, x, training):
        """
        get the learned embedding
        
        training specified to enable control over dropout
        """
        return self.encoder(x, training=training)

    def call(self, x, training=None):
        x_high,x_low = x
        emb_high_level = self.get_embedding(x_high, training=training)
        emb_low_level = self.get_embedding(x_low, training=training)
        task_prediction = self.fork(
            tf.concat([
                emb_high_level,
                emb_low_level
            ], axis=1),
            training=training
        )
        construct_high_level = self.decoder(emb_high_level, training=training)
        construct_low_level = self.decoder_low(emb_low_level, training=training)
        return construct_high_level, task_prediction, construct_low_level
