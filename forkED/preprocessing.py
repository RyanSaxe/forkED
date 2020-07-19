import pandas as pd
import numpy as np
import pickle
import os
import pdb

#create directory for processed data
path = "../data/ml-25m/processed"
try:
    os.mkdir(path)
    os.mkdir(os.path.join(path,'users'))
    os.mkdir(os.path.join(path,'maps'))
    os.mkdir(os.path.join(path,'movies'))
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

#open needed movielens dataset
fname = '../data/ml-25m/movies.csv'
movies = pd.read_csv(fname)

fname = '../data/ml-25m/ratings.csv'
ratings = pd.read_csv(fname)

fname = '../data/ml-25m/tags.csv'
tags = pd.read_csv(fname)

#create id mapping to ensure ids are in range (0,n)
n_movs = movies['movieId'].nunique()
n_users = ratings['userId'].nunique()
n_tags = tags['tag'].nunique()

movies_with_tags = tags['movieId'].unique()

movies = movies[movies['movieId'].isin(movies_with_tags)]
ratings = ratings[ratings['movieId'].isin(movies_with_tags)]

mov_map = dict(zip(movies['movieId'].unique(),range(n_movs)))
us_map = dict(zip(ratings['userId'].unique(),range(n_users)))
tag_map = dict(zip(tags['tag'].unique(),range(n_tags)))

#store mappings
with open('../data/ml-25m/processed/maps/movie.pkl','wb') as f:
    pickle.dump(mov_map,f)
with open('../data/ml-25m/processed/maps/user.pkl','wb') as f:
    pickle.dump(us_map,f)
with open('../data/ml-25m/processed/maps/tag.pkl','wb') as f:
    pickle.dump(tag_map,f)
#convert data to proper form for storage
ratings['user'] = ratings['userId'].map(us_map)
ratings['movie'] = ratings['movieId'].map(mov_map)
tags['tag'] = tags['tag'].map(tag_map)
tags['movie'] = tags['movieId'].map(mov_map)
#int to reduce space + store idx in same array
ratings['rating'] = ratings['rating'].apply(lambda x: 0 if x < 4 else 1).astype(int)

#this takes a while: get corresponding movies and ratings for users
#   probably a way to do this with one groupby?
user_mov = ratings.groupby('user')['movie'].apply(list)
user_rat = ratings.groupby('user')['rating'].apply(list)
movie_tag = tags.groupby('movie')['tag'].apply(list)
#process tags for each movie
for mid,tids in movie_tag.items():
    tag_ids = list(set(tids))
    values = np.zeros(len(tag_ids))
    for tag_id in tids:
        idx = tag_ids.index(tag_id)
        values[idx] += 1
    values = values/values.sum()
    arr = np.vstack([tag_ids,values])
    with open('../data/ml-25m/processed/movies/' + str(mid) + '.npy', 'wb') as f:
        np.save(f,arr)
#store one file per user for individual user-level batch loading
for uid,mids in user_mov.items():
    values = user_rat.loc[uid]
    arr = np.vstack([mids,values])
    with open('../data/ml-25m/processed/users/' + str(uid) + '.npy', 'wb') as f:
        np.save(f,arr)