import numpy as np
from forkED.generator import *
import sys
import os
import pickle

data_location = sys.argv[1]

movie_file = os.path.join(data_location, 'processed/maps/movie.pkl')
dim = max(pickle.load(
    open(movie_file, 'rb')
).values()) + 1
ddict = pickle.load(open(os.path.join(data_location,'processed/sparse_lookups/users.pkl'),'rb'))

for k,v in ddict.items():
    new_arr = np.ones_like(v[1])
    new_arr[np.where(v[1] < 4)] = -1
    ddict[k] = (v[0],new_arr)

dataset = AutoEncoderAugmentor(
    ddict,
    512,
    dim,
    verbose=False,
    file_cap_for_debug=1000,
)

from forkED.model import EncoderDecoder
from forkED.trainer import Trainer
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU') 
print('physical devices:',physical_devices)
#enable memory growth to avoid attempting to allocate
#full GPU memory. Grow allocated memory as needed.
tf.config.experimental.set_memory_growth(physical_devices[0], True)


model = EncoderDecoder(
    dim,
    256,
    32,
    3,
    dropout=0.2,
    out_act = tf.nn.tanh
)
model.loss = tf.losses.mean_squared_error
model.optimizer = tf.optimizers.Adam(lr=0.001)
t = Trainer(dataset, model)
t.train(25)