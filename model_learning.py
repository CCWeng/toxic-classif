import numpy as np
import pandas as pd
import xgboost as xgb
import sys


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

from matplotlib import pyplot as plt

import data_process as dp


target = 'toxic'


def target_encode(trn_series=None,
                  tst_series=None,
                  target=None,
                  min_samples_leaf=1,
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    trn_series : training categorical feature as a pd.Series
    tst_series : test categorical feature as a pd.Series
    target : target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior
    """
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


def my_scorer(estimatror, X, y):
	probs = estimatror.predict_proba(X)[:, 1]
	score = gini_normalized(y, probs)
	return score




def rmse_xgb(preds, dtrain):
    labels = dtrain.get_label()
    score = rmse(labels, preds)
    return [('rmse', score)]


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score


def xgbfit(alg, dtrain, dtest, predictors, 
	useTrainCV=True, 
	printFeatureImportance=True, 
	folds=None, cv_folds=5, 
	early_stopping_rounds=None):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], folds=folds, 
        	nfold=cv_folds, metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

        print "best n_estimators = %d" % cvresult.shape[0]
        print "CV Score (Train) : Mean - %.7g | Std - %.7g " % (cvresult.iloc[-1]['train-auc-mean'], cvresult.iloc[-1]['train-auc-std'])
        print "CV Score (Test) : Mean - %.7g | Std - %.7g " % (cvresult.iloc[-1]['test-auc-mean'], cvresult.iloc[-1]['test-auc-std'])

        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Fit log-target
    # log_target_trn = np.log1p(dtrain[target])
    # log_target_tst = np.log1p(dtest[target])

    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
        
    dtrain_pred = alg.predict(dtrain[predictors])
    dtest_pred = alg.predict(dtest[predictors])
    
    auc_trn = roc_auc_score(dtrain[target], dtrain_pred)
    auc_tst = roc_auc_score(dtest[target], dtest_pred)

    # auc_trn = roc_auc_score(log_target_trn, dtrain_pred)
    # auc_tst = roc_auc_score(log_target_tst, dtest_pred)
    # print "\nModel Report"
    print "auc score (Train): %f" % (auc_trn)
    print "auc score (Test): %f" % (auc_tst)

    if printFeatureImportance:
    	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    	feat_imp.plot(kind='bar', title='Feature Importances')
    	plt.ylabel('Feature Importance Score')
    	plt.show()

    return 


'''
==========
STACKING
==========
'''

def TrainAndValidationStacking(df_train, df_stack, df_val, models):

	sys.stdout.write("Stacking Train ... ")
	sys.stdout.flush()
	models, model_stack = TrainStacking(df_train, df_stack, models)
	sys.stdout.write("done.\n")

	sys.stdout.write("Stacking Predict ... ")
	sys.stdout.flush()
	p_val = PredictStacking(df_val, models, model_stack)
	sys.stdout.write("done.\n")

	y_val = df_val[target].values.astype(float)
	mae_val = mean_absolute_error(y_val, p_val)

	print 'MAE in validation :', mae_val

	return p_val, y_val, models, model_stack


def TrainStacking(df_train1, df_train2, models):	
	y_stack = df_train2[target].values.astype(float)

	X_stack = np.array([])
	for dt, mt, m, C in models:
		if dt == 'nume':
			X_train1 = df_train1[C].values.astype(float)
			X_train2 = df_train2[C].values.astype(float)
		else:
			X_train1 = df_train1[C].values
			X_train2 = df_train2[C].values
			X_train1, X_train2 = dp.CateEncoderTwoMat(X_train1, X_train2)

		y_train = df_train1[target].values.astype(float)
		if mt == 'classification':
			y_train = np.where(y_train > 0, 1, 0)	
		m.fit(X_train1, y_train)

		p_stack = m.predict(X_train2)
		p_stack = np.reshape(p_stack, (len(p_stack), 1))
		X_stack = np.hstack((X_stack, p_stack)) if X_stack.size else p_stack

	model_stack = LinearRegression()
	model_stack.fit(X_stack, y_stack)
	# stack_coef = model.coef_

	return models, model_stack


def PredictStacking(df, models, model_stack):
	X_stack = np.array([])
	for dt, mt, m, C in models:
		if dt == 'nume':
			X_test = df[C].values.astype(float)
		else:
			X_test = df[C].values
			X_test = dp.CateEncoderOneMat(X_test)

		p_stack = m.predict(X_test)
		p_stack = np.reshape(p_stack, (len(p_stack), 1))
		X_stack = np.hstack((X_stack, p_stack)) if X_stack.size else p_stack

	P = model_stack.predict(X_stack)

	return P








'''
======================
TRAIN AND VALIDATION
======================
'''


from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score


def validataion(trn, tst, features, targets, model):
	X_trn = trn[features]
	X_tst = tst[features]

	res = pd.DataFrame(index=targets, columns=['auc-trn', 'auc-tst'])
	for tgt in targets:
		print "\n==\nTarget : %s" % tgt

		y_trn = trn[tgt]
		y_tst = tst[tgt]

		model.fit(X_trn, y_trn)

		p_trn = model.predict(X_trn)
		p_tst = model.predict(X_tst)
		pp_trn = model.predict_proba(X_trn)[:, 1]
		pp_tst = model.predict_proba(X_tst)[:, 1]
		
		# acc_trn = accuracy_score(y_trn, p_trn)
		# acc_val = accuracy_score(y_val, p_val)
		# print acc_trn, acc_val

		rep_trn = classification_report(y_trn, p_trn)
		rep_tst = classification_report(y_tst, p_tst)
		
		auc_trn = roc_auc_score(y_trn, pp_trn)
		auc_tst = roc_auc_score(y_tst, pp_tst)
		
		print '* Train:'
		print rep_trn
		print "AUC = %.5f" % auc_trn

		print '* Test:'
		print rep_tst
		print "AUC = %.5f" % auc_tst

		res.loc[tgt, 'auc-trn'] = auc_trn
		res.loc[tgt, 'auc-tst'] = auc_tst

	return res




def EasyValidation(df_train, df_val, models, use_features):
	y_train = df_train['target'].values
	y_val = df_val['target'].values

	for name, model in models:
		print '--'
		print name
		
		model.fit(df_train[use_features], df_train['target'])
		
		p_train = model.predict(df_train[use_features])
		print 'Train :'
		print classification_report(y_train, p_train)
		p_train = model.predict_proba(df_train[use_features])
		print 'gini =', gini_normalized(y_train, p_train)

		p_val = model.predict(df_val[use_features])
		print 'Test :'
		print classification_report(y_val, p_val)
		p_val = model.predict_proba(df_val[use_features])
		print 'gini =', gini_normalized(y_val, p_val)

	return





def TrainAndValidation(df_train, df_val, model, columns, data_type='nume'):
	sys.stdout.write('Extract data ... ')
	sys.stdout.flush()

	if data_type == 'nume':
		X_train = df_train[columns].values.astype(float)
		X_val = df_val[columns].values.astype(float)
	else:
		X_train = df_train[columns].values
		X_val = df_val[columns].values
		X_train, X_val = dp.CateEncoderTwoMat(X_train, X_val)

	y_train = df_train[target].values.astype(float)
	y_val = df_val[target].values.astype(float)

	# y_train = abs(y_train)
	# y_val = abs(y_val)

	sys.stdout.write('done.\n')

	sys.stdout.write('Train ... ')
	sys.stdout.flush()

	model.fit(X_train, y_train)

	sys.stdout.write('done.\n')

	sys.stdout.write('Predict ...')
	sys.stdout.flush()
	
	p_train = model.predict(X_train)
	p_val = model.predict(X_val)

	sys.stdout.write('done.\n')

	mae_train = mean_absolute_error(y_train, p_train)
	mae_val = mean_absolute_error(y_val, p_val)

	print 'MAE in train :', mae_train
	print 'MAE in validation :', mae_val

	return y_train, y_val, p_train, p_val





def TrainAndValidationClassification(df_train, df_val, model, columns, data_type='nume'):
	sys.stdout.write('Extract data ... ')
	sys.stdout.flush()

	if data_type == 'nume':
		X_train = df_train[columns].values.astype(float)
		X_val = df_val[columns].values.astype(float)
	else:
		X_train = df_train[columns].values
		X_val = df_val[columns].values
		X_train, X_val = dp.CateEncoderTwoMat(X_train, X_val)

	target = 'logerror_sign'
	y_train = df_train[target].values.astype(float)
	y_val = df_val[target].values.astype(float)

	# y_train = abs(y_train)
	# y_val = abs(y_val)

	sys.stdout.write('done.\n')

	sys.stdout.write('Train ... ')
	sys.stdout.flush()

	model.fit(X_train, y_train)

	sys.stdout.write('done.\n')

	sys.stdout.write('Predict ...')
	sys.stdout.flush()
	
	p_train = model.predict(X_train)
	p_val = model.predict(X_val)

	sys.stdout.write('done.\n')

	acc_train = (p_train == y_train).sum() / float(len(y_train))
	acc_val = (p_val == y_val).sum() / float(len(y_val))

	print 'Accuracy in train :', acc_train
	print 'Accuracy in validation :', acc_val

	return y_train, y_val, p_train, p_val




'''
============
EVALUATION
============
'''


def rmsle(y_true, y_pred):
	# return np.sqrt(np.mean((np.log(y_pred+1) - np.log(y_true+1)) ** 2))
	return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def rmse(y_true, y_pred):
	return np.sqrt(np.mean((y_true - y_pred) ** 2))

def rss(y_true, y_pred):
	return sum((y_true - y_pred) ** 2)


def my_scorer(estimatror, X, y):
	preds = estimatror.predict(X)
	score = rmse(y, preds)
	return score


def modelfit(alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
        
    #Predict training set:
    dtrain_pred = alg.predict(dtrain[predictors])
    dtest_pred = alg.predict(dtest[predictors])
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=cv_folds, scoring=my_scorer)
    
    #Print model report:
    print "\nModel Report"

    rmse_trn = rmse(dtrain[target], dtrain_pred)
    rmse_tst = rmse(dtest[target], dtest_pred)
    print "rmse (Train): %f, (Test) %f" % (rmse_trn, rmse_tst)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(np.abs(alg.coef_), predictors).sort_values(ascending=False)
        feat_imp[:50].plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()




