# forkED

Novel Encoder Decoder architecture for end-to-end learning of embeddings with hierarchical components

forkED stands for fork encoder decoder. The concept is as follows:

- The input describes a hierarchical relationship. For example, the movielens dataset input is a user, and a user is described by their movie ratings. This algorithm learns a shared encoder for representing both users and movies.
- The latent encoding is then "forked" into multiple processes, hence the name forkED. These processes are designed according to the problem at hand, but generally they fall into the following set: 
  - Reconstruct original input (high hierarchy level --- users)
  - Construct some information from subset of original input (low hierarchy level --- movies -> tags of movies)
  - Some prediction task that uses both high and low level context (given a movie and a user, what is the rating).
  
 ## Data
 
 Currently I am developing this on the movielens dataset. After cloning this repository, create a folder `data`. Download the ml-25m research stable dataset from [here](https://grouplens.org/datasets/movielens/), and place the dataset in the `data` folder. Then, `python preprocessing.py` will create the needed subdirectories and files for proper batch training.
 
 ## More Detailed Description Coming Soon
