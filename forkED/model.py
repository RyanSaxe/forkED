from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

class MLP_3(Model):
    def __init__(
        self,
        dims,
        out_dim,
        hidden_act='relu',
        out_act='sigmoid',
        noise=None,
        bottleneck_flag=True,
        reg = [None, None, None],
        name=None,
    ):
        super().__init__(name=name)
        if bottleneck_flag:
            assert dims[-1] >= out_dim, (
                f'The output dim must be smaller than {dims[-1]}'
            )
        else:
            assert dims[-1] <= out_dim, (
                f'The output dim must be larger than {dims[-1]}'
            )

        self.noise = None if noise is None else Dropout(noise)
        self.drop_1 = None if reg[0] is None else Dropout(reg[0])
        self.drop_2 = None if reg[1] is None else Dropout(reg[1])
        self.drop_3 = None if reg[2] is None else Dropout(reg[2])

        self.layer_1 = Dense(dims[0], activation=hidden_act)
        self.layer_2 = Dense(dims[1], activation=hidden_act)
        self.layer_3 = Dense(dims[2], activation=hidden_act)
        self.out = Dense(out_dim, activation=out_act)

    def call(self, x, skip_noise=False, training=None):
        #may want to skip input noise for parts of the fork
        #during training, so need additional flag for no dropout
        if (self.noise is not None) and (not skip_noise):
            x = self.noise(x)
        x = self.layer_1(x)
        if self.drop_1 is not None:
            x = self.drop_1(x)
        x = self.layer_2(x)
        if self.drop_2 is not None:
            x = self.drop_2(x)
        x = self.layer_3(x)
        if self.drop_3 is not None:
            x = self.drop_3(x)
        return self.out(x)

class ForkEncoderDecoder(Model):
    """
    input_dim:      dimension of the input data
    first_hidden_dim:   dimension of the first hidden layer of the encoder
    latent_dim:         dimension of the bottleneck of the encoder
    fork_task_dim:      dimension of the output of the fork task
    dlow_dim:           dimension of the output of the lower dim construct
                            (high_dim construct always = input_dim)
    fork_task_act:      activation for output of fork task
    dhigh_act:          activation for output of high dim task 
    dlow_act:           activation for output of low dim task
    """
    def __init__(
        self,
        input_dim,
        first_hidden_dim,
        latent_dim,
        fork_task_dim,
        dlow_dim,
        fork_task_act='sigmoid',
        dhigh_act='sigmoid',
        dlow_act='sigmoid',
        name=None,
    ):
        super().__init__(name=name)
        construct_dims = [first_hidden_dim // (2 ** i) for i in range(3)]
        self.encoder = MLP_3(
            construct_dims,
            latent_dim,
            out_act='relu',
            reg=[0.4,0.2,0.2],
            name='encoder'
        )
        self.decoder_high = MLP_3(
            construct_dims[::-1],
            input_dim,
            bottleneck_flag=False,
            out_act=dhigh_act,
            name='decoder_high'
        )
        self.decoder_low = MLP_3(
            construct_dims[::-1],
            dlow_dim,
            bottleneck_flag=False,
            out_act=dlow_act,
            name='decoder_low'
        )
        fork_dims = [latent_dim // (2 ** i) for i in range(1,4)]
        self.fork = MLP_3(
            fork_dims,
            fork_task_dim,
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
            ], axis=1)
        )
        construct_high_level = self.decoder_high(emb_high_level, training=training)
        construct_low_level = self.decoder_low(emb_low_level, training=training)
        return construct_high_level, task_prediction, construct_low_level
