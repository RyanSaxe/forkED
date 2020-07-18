# forkED

Custom DNN architecture for end-to-end learning of embeddings within a specific hierarchy.

forkED stands for fork encoder decoder. The concept is as follows:

1. The input describes a hierarchical relationship. For example, the movielens dataset input is a user, and a user is described by their movie ratings. This algorithm learns a shared encoder for representing both users and movies.
2. The latent encoding is then "forked" into multiple processes, hence the name forkED. These processes are designed according to the dataset, but generally they fall into the following set: reconstruct original input (high hierarchy level --- users), reconstruct subset of original input (low hierarchy level --- movies), and some prediction task that uses both high and low level context (given a movie and a user, what is the rating).
