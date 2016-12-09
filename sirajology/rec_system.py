import numpy as np
import scipy as sp
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

# print(repr(data['train']))
# print(repr(data['test']))

model = LightFM(loss='warp')
model.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendations(model, data, user_ids):
    n_users, n_items = data['train'].shape
    for user in user_ids:
        known_positives = data['item_labels'][data['train'].tocsr()[user].indices]
        scores = model.predict(user, np.arange(n_items))
        top_items = data['item_labels'][np.argsort(-scores)]
        # print out the results
        print("User %s\n Known positives: " % user)
        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")
        for x in top_items[:3]:
            print("        %s" % x)

sample_recommendations(model, data, [3, 25, 450])