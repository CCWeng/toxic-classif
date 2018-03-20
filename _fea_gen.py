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


# trn, tst = train_test_split(trn, train_size=0.35, test_size=0.15)
# trn, tst = train_test_split(trn, train_size=0.15, test_size=0.08)


#===
min_df = 0.001
tok = r'(?u)\b\w*[a-zA-Z]\w*\b'


# from sklearn.model_selection import cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import make_scorer
# from sklearn.metrics import roc_auc_score
# from sklearn.grid_search import GridSearchCV
# import time


# tfidf_vec = TfidfVectorizer(
# 	min_df=min_df, 
# 	token_pattern=tok, 
# 	analyzer='char',
# 	strip_accents='unicode',
# 	ngram_range=(2, 2),
# 	sublinear_tf=True,
# 	stop_words='english'
# 	)

# estimators = list()
# estimators.append(('vectr', tfidf_vec))
# estimators.append(('lr', LogisticRegression()))
# pipe = Pipeline(estimators)


# param_grid = dict()
# # param_grid['vectr__strip_accents'] = ['ascii', 'unicode', None]
# # param_grid['vectr__sublinear_tf'] = [False, True]
# param_grid['vectr__ngram_range'] = [(i, i) for i in range(2, 7)]
# # param_grid['vectr__stop_words'] = ['english', None]



# gsearch = GridSearchCV(
# 	estimator=pipe,
# 	param_grid=param_grid,
# 	scoring=make_scorer(roc_auc_score),
# 	iid=False,
# 	cv=5)


# start_time = time.time()
# gsearch.fit(trn['comment_text'], trn['toxic'])
# elapsed_time = time.time() - start_time
# print elapsed_time


# gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_


word_vectorizer = TfidfVectorizer(
	min_df=min_df, 
	token_pattern=tok, 
	analyzer='word',
	strip_accents='unicode',
	sublinear_tf=False,
	ngram_range=(1, 1), 
	max_features=10000 )

trn, tst, word_tfidf_columns = fg2.CreateTfidfFeaturess(trn, tst, word_vectorizer, postfix='_word')


# char_vectorizer = TfidfVectorizer(
# 	min_df=min_df,
# 	token_pattern=tok,
# 	analyzer='char',
# 	strip_accents='unicode',
# 	sublinear_tf=True,
# 	ngram_range=(3, 3),
# 	stop_words='english', 
# 	max_features=50000)

# trn, tst, char_tfidf_columns = fg2.CreateTfidfFeaturess(trn, tst, char_vectorizer, postfix='_char')



trn, tst, tfidf_lr_columns = fg2.CreateTfidfLogisticRegColumns(trn, tst)



trn['word_count'] = trn.comment_text.apply(fg2.count_word)
tst['word_count'] = tst.comment_text.apply(fg2.count_word)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(trn[['word_count']])
trn['word_count'] = scaler.transform(trn[['word_count']])
tst['word_count'] = scaler.transform(tst[['word_count']])


# trn.loc[trn.word_count > 4, 'word_count'] = 4
# tst.loc[tst.word_count > 4, 'word_count'] = 4



wcount_columns = ['word_count']


bag_of_words, _ = fa.GetBagOfWords(trn, word_tfidf_columns)
trn, tst, word_exist_columns = fg2.CreateWordExistFeatures(trn, tst, bag_of_words)




oof_columns = list()
for tgt in targets:
	print "Create OOF columns for %s" % tgt
	trn, tst, oof = fg.CreateOOFColumns(trn, tst, word_exist_columns, tgt_c=tgt)
	oof_columns += oof


smooth_columns = list()
for tgt in targets:
	print "Create smoothing columns for %s... " % tgt
	trn, tst, smooth = fg.CreateSmoothingColumns(trn, tst, word_exist_columns, tgt_c=tgt)
	smooth_columns += smooth




# Create features from fastText
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


trn, tst, ft_lb_columns, ft_pr_columns = fg2.CreateFastTextColumns(trn, tst, ft_params)


use_columns = list()
use_columns += word_tfidf_columns
# use_columns += char_tfidf_columns
use_columns += tfidf_lr_columns

use_columns += wcount_columns
use_columns += oof_columns
use_columns += smooth_columns
use_columns += ft_pr_columns





use_columns2 = tfidf_lr_columns + wcount_columns + oof_columns + smooth_columns + ft_pr_columns


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()


print "train data size :", trn.shape
print "test data size :", tst.shape


print "== Train & Predict =="


## Logistic Regression
scores = []
submission = pd.DataFrame.from_dict({'id': tst['id']})
for tgt in targets:
	trn[tgt] = trn[tgt]
	classifier = LogisticRegression(C=0.1, solver='sag')
	cv_score = np.mean(cross_val_score(classifier, trn[use_columns], trn[tgt], cv=3, scoring='roc_auc'))
	scores.append(cv_score)
	print('CV score for class {} is {}'.format(tgt, cv_score))
	classifier.fit(trn[use_columns], trn[tgt])
	submission[tgt] = classifier.predict_proba(tst[use_columns])[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

submission.to_csv('__output/submission_lr.csv', index=False)


## XGBoost


# from xgboost.sklearn import XGBClassifier

# xgb4 = XGBClassifier(
# 	learning_rate =0.01,
# 	n_estimators=5000,
# 	max_depth=3,
# 	min_child_weight=1,
# 	gamma=0.2,
# 	subsample=0.8,
# 	colsample_bytree=0.8,
# 	reg_alpha=0.001,
# 	objective= 'binary:logistic',
# 	nthread=4,
# 	scale_pos_weight=1,
# 	seed=27 )


# xgb5 = XGBClassifier(
# 	learning_rate =0.01,
# 	n_estimators=5000,
# 	max_depth=4,
# 	min_child_weight=6,
# 	gamma=0,
# 	subsample=0.8,
# 	colsample_bytree=0.8,
# 	reg_alpha=0.005,
# 	objective= 'binary:logistic',
# 	nthread=4,
# 	scale_pos_weight=1,
# 	seed=27 )


# scores = []
# submission = pd.DataFrame.from_dict({'id': tst['id']})
# for tgt in targets:
# 	trn[tgt] = trn[tgt]
# 	classifier = xgb4
# 	classifier.fit(trn[use_columns], trn[tgt])
# 	submission[tgt] = 1 - classifier.predict_proba(tst[use_columns])[:, 0]

# print('Total CV score is {}'.format(np.mean(scores)))

# submission.to_csv('__output/submission_xgb4.csv', index=False)













