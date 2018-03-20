# ft_params = dict()
# for tgt in targets:
# 	best_p_param, best_r_param = fa.GridSearchFastText(ft_files[tgt]['trn'], ft_files[tgt]['tst'], params)
# 	ft_params[tgt] = best_p_param




ft_params=dict()
ft_params['toxic'] = {'epoch': 5, 'lr': 1.0, 'word_ngrams': 1}
ft_params['severe_toxic'] = {'epoch': 1, 'lr': 0.4, 'word_ngrams': 2}
ft_params['obscene'] = {'epoch': 25, 'lr': 0.1, 'word_ngrams': 1}
ft_params['threat'] = {'epoch': 15, 'lr': 0.4, 'word_ngrams': 2}
ft_params['insult'] = {'epoch': 25, 'lr': 0.1, 'word_ngrams': 2}
ft_params['identity_hate'] = {'epoch': 25, 'lr': 0.7, 'word_ngrams': 2}




for tgt in targets:
	tst[tgt] = 0

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

