import pandas as pd
import numpy as np
import pickle
import os

#create directory for processed data
path = "../data/ml-25m/processed"
try:
    os.mkdir(path)
    os.mkdir(os.path.join(path,'users'))
    os.mkdir(os.path.join(path,'maps'))
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

#open needed movielens dataset
fname = '../data/ml-25m/movies.csv'
movies = pd.read_csv(fname)

fname = '../data/ml-25m/ratings.csv'
ratings = pd.read_csv(fname)

#create id mapping to ensure ids are in range (0,n)
n_movs = movies['movieId'].nunique()
n_users = ratings['userId'].nunique()

mov_map = dict(zip(movies['movieId'].unique(),range(n_movs)))
us_map = dict(zip(ratings['userId'].unique(),range(n_users)))

#store mappings
with open('../data/ml-25m/processed/maps/movie.pkl','wb') as f:
    pickle.dump(mov_map,f)

with open('../data/ml-25m/processed/maps/user.pkl','wb') as f:
    pickle.dump(us_map,f)

#convert data to proper form for storage
ratings['user'] = ratings['userId'].map(us_map)
ratings['movie'] = ratings['movieId'].map(mov_map)
#int to reduce space + store idx in same array
ratings['rating'] = (ratings['rating'] * 10).astype(int)

#this takes a while: get corresponding movies and ratings for users
#   probably a way to do this with one groupby?
user_mov = ratings.groupby('user')['movie'].apply(list)
user_rat = ratings.groupby('user')['rating'].apply(list)

#store one file per user for individual user-level batch loading
for uid,mids in user_mov.items():
    values = user_rat.loc[uid]
    arr = np.vstack([mids,values])
    with open('../data/ml-25m/processed/users/' + str(uid) + '.npy', 'wb') as f:
        np.save(f,arr)

#add processing of movie features (e.g. tags)