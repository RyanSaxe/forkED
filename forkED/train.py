from model import ForkEncoderDecoder
from generator import ProngGenerator
import pickle
import numpy as np
import os

map_files = '../data/ml-25m/processed/maps'
data_f = '../data/ml-25m/processed/users'

print('Loading Mapping Files')
umap = pickle.load(open(map_files + "/user.pkl",'rb'))
mmap = pickle.load(open(map_files + "/movie.pkl",'rb'))

n_users = len(umap)
n_movs = len(mmap)

train_users = np.random.choice(list(range(n_users)), int(n_users * 0.8), replace=False)
val_users = list(set(list(range(n_users))) - set(train_users))

print('Initializing Generators')
training_generator = ProngGenerator(
    train_users,
    n_movs,
    data_f,
    batch_size=256
)

validation_generator = ProngGenerator(
    val_users,
    n_movs,
    data_f,
    batch_size=256
)

print('Compiling Model')
m = ForkEncoderDecoder(
    n_movs,
    256,
    32,
    1,
    n_movs
)

m.compile(
    optimizer='adam',
    loss=['binary_crossentropy','mean_squared_error', 'binary_crossentropy'],
    loss_weights=[1.0,1.0,1.0],
    metrics=['mean_absolute_error'],
)

print('Fitting Model')
m.fit_generator(
    epochs=5,
    generator=training_generator,
    validation_data=validation_generator,
)