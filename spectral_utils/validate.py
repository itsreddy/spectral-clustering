import itertools
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
import numpy as np

def predict(validate_df, train_cid,
                    train_mid, train_ratings, train_movie_by_movie, 
                    cluster_labels, label_map, block_size=10000):

    def fill_per_window(args):                   
        window_x, _ = args
        tmp = np.ctypeslib.as_array(shared_array)

        for i in range(window_x, window_x + block_size):
            record = validate_df.iloc[i]
            cid = train_cid.get(record[1])
            mid = train_mid.get(record[0])

            if mid is not None:
                if cid is not None:
                    cluster_id = cluster_labels[mid]
                    neighbours = []
                    cluster_set = label_map.get(cluster_id)
                    for j in cluster_set:
                        if j != mid and train_ratings[cid][j] != 0:
                            similarity = train_movie_by_movie[j][mid]
                            neighbours.append([j, similarity, train_ratings[cid][j]])

                    # try:    
                    if len(neighbours) > 10:    
                        neighbours.sort(key = lambda x: x[1])
                        l = len(neighbours)
                        nearest_neighbours = np.array(neighbours[l-10:])
                        nn_ratings = nearest_neighbours[:, 2]
                        nn_weight = nearest_neighbours[:, 1]
                        rating_prediction = np.average(nn_ratings, weights=nn_weight)
                    else:
                        if len(neighbours) > 1:
                            movie_r = np.array(neighbours)[:, 2]
                            nonzeros = movie_r[np.nonzero(movie_r)]
                            rating_prediction = np.mean(nonzeros)
                        else:
                            movie_r = train_ratings[:, mid]
                            nonzeros = movie_r[np.nonzero(movie_r)]
                            rating_prediction = np.mean(nonzeros)
                    # except:
                    #     print(record, neighbours)

                else:
                    movie_r = train_ratings[:, mid]
                    nonzeros = movie_r[np.nonzero(movie_r)]
                    rating_prediction = np.mean(nonzeros)
            else:
                if cid is not None:
                    movie_r = train_ratings[cid]
                    nonzeros = movie_r[np.nonzero(movie_r)]
                    rating_prediction = np.mean(nonzeros)
                else:
                    rating_prediction = 3

            # pred_record = [record[0], record[1], rating_prediction, round(rating_prediction), record[2]]
            # m_preds.append(pred_record)

            if rating_prediction <= 0.5:
                rating_prediction = 1
            if rating_prediction > 5.5:
                rating_prediction = 5 

            tmp[i][0] = record[0]
            tmp[i][1] = record[1]
            tmp[i][2] = rating_prediction
            # tmp[i][3] = record[2]

        print(window_x + block_size , ' records done')

    size = validate_df.shape[0]

    result = np.ctypeslib.as_ctypes(np.zeros((size, 3)))
    shared_array = sharedctypes.RawArray(result._type_, result)

    window_idxs = [(i, 0) for i in range(0, size, block_size)]

    p = Pool(10)
    res = p.map(fill_per_window, window_idxs)
    result = np.ctypeslib.as_array(shared_array)

    return result