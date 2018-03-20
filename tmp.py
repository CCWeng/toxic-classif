import pandas as pd
import numpy as np
import pdb
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import feature_generation as fg
import feature_generation_text as fg2
import feature_analysis as fa
import model_learning as ml


trn = pd.read_csv("__input/train.csv")
tst = pd.read_csv("__input/test.csv")
sub = pd.read_csv("__input/sample_submission.csv")



targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


trn, tst = train_test_split(trn, train_size=0.15, test_size=0.08)


import data_process as dp

ft_files = dp.CreateFasttextFiles(trn, tst, '__fasttext')

params = dict()
params['word_ngrams'] = [1, 2, 3]
params['epoch'] = [1, 5, 15, 25]
params['lr'] = [0.1, 0.4, 0.7, 1.0]

ft_params = dict()
for tgt in targets:
	best_p_param, best_r_param = fa.GridSearchFastText(ft_files[tgt]['trn'], ft_files[tgt]['tst'], params)
	ft_params[tgt] = best_p_param


# trn, tst, ft_lb_columns, ft_pr_columns = fg2.CreateFastTextColumns(trn, tst, ft_params)

