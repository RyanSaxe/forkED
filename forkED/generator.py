import numpy as np
import tensorflow as tf
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class Augmentation:
    """
    Base class for data augmentation process for solving any problem on collections.

    An example of a collection is a Netflix User. A user has seen and rated movies, and
        so this "collection" is a vector of dimension = the number of movies on nextflix,
        where the value at index i in this vector is the users rating for movie i

    Parameters:
    _______________

        read_loc: the file location containing .npy files for collections such that
                        idxs,vals = np.load(xxx.npy) gets the indices and values 
    
        batch_size: parameter for how many collections to process per batch

        dim: the total number of items in the dataset (e.g. movies on netflix)

        repeat: the number of times to repeat the dataset

        noise: a tuple such that noise[0] is the percentage of items to remove and
                                noise[1] is the percentage of items to add

        proba_loc: the location for a .npy file that is a vector of length `dim` such that
                    the ith element corresponds to the probability for negatively sampling
                    the ith item

    Usage:
    __________________

        >>> generator = Augmentation(
                '/.../path/to/data/',
                batch_size = 256,
                dim = 1000,
            )
        >>> while generator.has_next_batch():
                batch = generator.load_batch()
                # Do something with the batch
    """
    def __init__(
        self,
        read_loc,
        batch_size,
        dim,
        repeat = 0,
        noise = (0.1, 0.1),
        proba_loc = None,
    ):
        self.read_loc = read_loc
        self.batch_size = batch_size
        self.dim = dim
        self.repeat = repeat
        self.noise_by_addition = noise[0]
        self.noise_by_removal = noise[1]
        if proba_loc is None:
            self.neg_sampler = None
        else:
            self.neg_sampler = np.load(proba_loc)
        all_files = [os.path.join(self.read_loc,x) for x in os.listdir(self.read_loc) if x.endswith('.npy')]
        self.files = all_files[:]
        for repetition in range(repeat):
            self.files += all_files
        self.final_batch = len(self.files)//self.batch_size + (len(self.files) % self.batch_size > 0)
        self.initialize()

    @property
    def has_next_batch(self):
        """
        Determine if there is another data batch to get

        :return: Boolean
        """
        return self.cur_batch < self.final_batch

    def __call__(self, input_file):
        """
        concept for multithreaded taken from: https://github.com/mdbloice/Augmentor/

            Function used by the ThreadPoolExecutor to process the pipeline
            using multiple threads. Do not call directly.
            This function does nothing except call :func:`_load_batch`, rather
            than :func:`_load_batch` being called directly in :func:`load_batch`.
            This makes it possible for the procedure to be *pickled* and
            therefore suitable for multi-threading.

        :param batch: The file locations for current data batch
        :return: result from :func: `_load_batch`
        """
        return self._augment_file(input_file)

    def initialize(self):
        self.cur_batch = 0
        np.random.shuffle(self.files)

    def load_batch(self, multi_threaded=True):
        """
        concept for multithreaded taken from: https://github.com/mdbloice/Augmentor/

        Load a batch of data and apply augmentation

        :param multi_threaded: Boolean to determine whether to use multithreaded process
        :return: Tensor that is an augmented data batch
        """
        batch_files = self.files[
            self.batch_size * self.cur_batch : self.batch_size * (1 + self.cur_batch)
        ]
        batch = []
        if multi_threaded:
            with tqdm(total=len(batch_files), desc="Augmenting Data", unit="Samples") as progress_bar:
                with ThreadPoolExecutor(max_workers=None) as executor:
                    for result in executor.map(self, batch_files):
                        batch.append(result)
                        progress_bar.update(1)
        else:
            with tqdm(total=len(batch_files), desc="Augmenting Data", unit="Samples") as progress_bar:
                for input_file in batch_files:
                    result = self._augment_file(input_file)
                    batch.append(result)
                    progress_bar.update(1)
        self.cur_batch += 1
        return tf.convert_to_tensor(np.vstack(batch),tf.float32)

    def _augment_file(self, input_file):
        output = np.zeros(self.dim)
        idxs,vals = np.load(input_file)
        np.put(output,idxs,vals)
        return self.apply_augmentation(output)
    
    def apply_augmentation(self, array):
        includes = np.where(array != 0)[0]
        size = len(includes)
        if self.noise_by_addition > 0:
            excludes = np.where(array == 0)[0]
            n_neg_samples = np.random.binomial(size=size, n=1, p=self.noise_by_addition).sum()
            if self.neg_sampler:
                neg_sampler = self.neg_sampler[excludes]/self.neg_sampler[excludes].sum()
                addition_mask = np.random.choice(
                    excludes,
                    size=n_neg_samples,
                    p=neg_sampler,
                    replace=False
                )
            else:
                addition_indices = np.random.choice(
                    excludes,
                    size=n_neg_samples,
                    replace=False
                )
            addition_vals = self.get_reasonable_addition_noise(addition_indices)
            np.put(array, addition_indices, addition_vals)
        if self.noise_by_removal > 0:
            size = len(includes)
            removal_mask = np.random.choice(
                a=[True, False], 
                size=size, 
                p=[self.noise_by_removal, 1-self.noise_by_removal]
            )
            removal_indices = includes[removal_mask]
            np.put(array, removal_indices, 0)
        return array

    def get_reasonable_addition_noise(self, indices):
        """
        If the vectors are not binary, overwrite this function
        """
        return np.ones(indices.shape)
