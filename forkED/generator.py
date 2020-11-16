import numpy as np
import tensorflow as tf
from forkED.base import BaseGenerator

class AutoEncoderAugmentor(BaseGenerator):
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

        file_cap_for_debug: Upper limit on number of files to open for quicker debugging

    Usage:
    __________________

        >>> generator = Augmentation(
                '/.../path/to/data/',
                batch_size = 256,
                dim = 1000,
            )
        >>> while generator.has_next_batch:
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
        verbose = True,
        file_cap_for_debug = None,
        storage_flag = True,
    ):
        super().__init__(
            read_loc,
            batch_size,
            dim,
            repeat = repeat,
            verbose = verbose,
            file_cap_for_debug = file_cap_for_debug,
            storage_flag = storage_flag
        )
        self.noise_by_addition = noise[0]
        self.noise_by_removal = noise[1]

        if proba_loc is None:
            self.neg_sampler = None
        else:
            self.neg_sampler = np.load(proba_loc)
    
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
