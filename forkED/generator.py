import numpy as np
import tensorflow.keras as keras
import os

class ProngGenerator(keras.utils.Sequence):
    """
    Data Generator skeleton taken from:
        https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(
        self,
        list_IDs,
        n_cols,
        data_dir,
        secondary_out_size = None,
        batch_size=32,
        shuffle=True
    ):
        self.list_IDs = list_IDs
        self.sec_dim = n_cols if secondary_out_size is None else secondary_out_size
        self.n_cols = n_cols
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.n_cols))
        X_one_hot = np.zeros((self.batch_size, self.n_cols))
        X_proba = np.zeros((self.batch_size,self.sec_dim))

        y = np.zeros((self.batch_size, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            #get ids and ratings
            idxs,vals = np.load(os.path.join(self.data_dir,'users', str(ID) + '.npy'))
            vals = vals/50.0
            X[i,idxs] = vals
            #mask a rating for secondary target
            mask_and_target = np.random.choice(len(idxs))
            X[i,idxs[mask_and_target]] = 0
            X_one_hot[i,idxs[mask_and_target]] = 1
            tag_idxs,tag_vals = np.load(os.path.join(self.data_dir, 'movies',str(idxs[mask_and_target]) + '.npy'))
            tag_idxs = tag_idxs.astype(int)
            X_proba[i, tag_idxs] = tag_vals
            # store secondary target
            y[i] = vals[mask_and_target]

        return [X, X_one_hot], [X, y, X_proba]