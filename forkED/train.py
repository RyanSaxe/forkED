from model import ForkEncoderDecoder
from losses import nonzero_MAE
from generator import ProngGenerator
import pickle
import numpy as np
import os
import tensorflow as tf

print(tf.__version__)

map_files = '../data/ml-25m/processed/maps'
data_f = '../data/ml-25m/processed/'

print('Loading Mapping Files')
umap = pickle.load(open(map_files + "/user.pkl",'rb'))
mmap = pickle.load(open(map_files + "/movie.pkl",'rb'))
tmap = pickle.load(open(map_files + "/tag.pkl",'rb'))

n_users = len(umap)
n_movs = len(mmap)
n_tags = len(tmap)

print(n_users,n_movs,n_tags)

train_users = np.random.choice(list(range(n_users)), int(n_users * 0.01), replace=False)
val_users = list(set(list(range(n_users))) - set(train_users))[:500]

print('Initializing Generators')
training_generator = ProngGenerator(
    train_users,
    n_movs,
    data_f,
    secondary_out_size=n_tags,
    batch_size=256
)

validation_generator = ProngGenerator(
    val_users,
    n_movs,
    data_f,
    secondary_out_size=n_tags,
    batch_size=256
)

print('Compiling Model')
m = ForkEncoderDecoder(
    n_movs,
    256,
    32,
    1,
    n_tags,
    dlow_act='softmax'
)

m.compile(
    optimizer='adam',
    loss=['binary_crossentropy','binary_crossentropy', 'kullback_leibler_divergence'],
    loss_weights=[0.1,1.0,0.1],
    metrics=[nonzero_MAE],
)

print('Fitting Model')
m.fit_generator(
    epochs=5,
    generator=training_generator,
    validation_data=validation_generator,
)