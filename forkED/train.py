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

train_users = np.random.choice(list(range(n_users)), int(n_users * 0.1), replace=False)
# val_users = list(set(list(range(n_users))) - set(train_users))[:500]
test_options = list(set(list(range(n_users))) - set(train_users))
#test on 10,000 out of sample users
test_users = np.random.choice(test_options, 10000, replace=False)

batch_size = 256

print('Initializing Generators')
training_generator = ProngGenerator(
    train_users,
    n_movs,
    data_f,
    secondary_out_size=n_tags,
    batch_size=batch_size
)

# validation_generator = ProngGenerator(
#     val_users,
#     n_movs,
#     data_f,
#     secondary_out_size=n_tags,
#     batch_size=batch_size
# )

test_generator = ProngGenerator(
    test_users,
    n_movs,
    data_f,
    secondary_out_size=n_tags,
    batch_size=batch_size
)

print('Compiling Model')
m = ForkEncoderDecoder(
    n_movs,
    batch_size,
    32,
    1,
    n_tags,
    dlow_act='softmax'
)

m.compile(
    optimizer='adam',
    loss=['binary_crossentropy','binary_crossentropy', 'kullback_leibler_divergence'],
    loss_weights=[0.1,1.0,0.1],
    metrics=[nonzero_MAE,'accuracy'],
)

print('Fitting Model')
m.fit(
    training_generator,
    epochs=50,
    batch_size=batch_size,
    #validation_data=validation_generator,
)
print('Testing Model')
resacc = np.zeros(batch_size)
resrmse = np.zeros(batch_size)
acc = tf.keras.metrics.BinaryAccuracy()
rmse = tf.keras.metrics.RootMeanSquaredError()

for i in range(batch_size):
    print('batch',i,'/',batch_size)
    testX,testY = test_generator[i]
    predictions = m.predict(testX)
    we_care_about = predictions[1]
    acc.update_state(testY[1],we_care_about)
    rmse.update_state(testY[1],we_care_about)
    resacc[i] = acc.result().numpy()
    resrmse[i] = rmse.result().numpy()
    acc.reset_states()
    rmse.reset_states()
print('Binary Accuracy:',resacc.mean())
print('RMSE:',resrmse.mean())