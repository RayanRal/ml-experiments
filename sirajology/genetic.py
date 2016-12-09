from tpot import TPOT
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

telescope = pd.read_csv('telescope.csv')

telescope_shuffle = telescope.iloc[np.random.permutation(len(telescope))]
tele = telescope_shuffle.reset_index(drop=True)


tele['Class'] = tele['Class'].map({'g': 0, 'h': 1})
tele_class = tele['Class'].values

train_indices, validation_indices = train_indices, test_indices = train_test_split(tele.index, stratify=tele_class,
                                                                                   train_size=0.75, test_size=0.25)

tpot = TPOT(generations=5, verbosity=2)


