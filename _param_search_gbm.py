from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import pickle as pkl

import model_learning as ml



# Set data
use_features = list()
use_features += nume_features
use_features += bin_features
use_features += dumm_features
use_features += inv_features
use_features += poly_features



df_cv, _ = train_test_split(df_train, train_size=0.3)


X_trn = df_cv[use_features]
y_trn = df_cv['target']

# X_tst = df_val2[use_features]
# y_tst = df_val2['target']


# Set model
model_gbc = GradientBoostingClassifier()


estimators = list()
estimators.append(('standard scaler', StandardScaler()))
estimators.append(('gradient boosting classifier', model_gbc))

model_pipe = Pipelien(estimators)



# Set parameter grids 
parameters = dict()
parameters['learning_rate'] = np.arange(0.01, 0.1, 0.01).tolist() + np.arange(0.1, 1, 0.1).tolist()
parameters['n_estimators'] = range(100, 1200, 100)
parameters['max_depth'] = range(6, 15, 2)
parameters['min_samples_leaf'] = range(6, 15, 2)
parameters['subsample'] = np.arange(0.4, 1.1, 0.2)
parameters['max_features'] = np.arange(0.4, 1.1, 0.2)



# Run grid search
def my_scorer(estimatror, X, y):
	probs = estimatror.predict_proba(X)[:, 1]
	score = ml.gini_normalized(y, probs)
	return score


grid = GridSearchCV(
	estimator=model_pipe, 
	param_grid=parameters, 
	scoring=my_scorer, 
	cv=None,
	n_jobs=-1,
	verbose=10 )


grid.fit(X_trn, y_trn)


# Print & save best estimator, paramters and score
fname = 'best_model_gbc.pkl'
fhandle = open(fname, 'wb')
pkl.dump(grid.best_estimator_, fhandle)

print grid.best_params_
print grid.best_score_






'''
==========
Ref - 01
==========
'''

def modelfit(alg, dtrain, dtest, predictors, performCV=True, printFeatureImportance=True, cv_folds=5):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['target'])
        
    #Predict training set:
    # dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain['target'], cv=cv_folds, scoring=my_scorer)
        # cv_score = cross_val_score(alg, dtrain[predictors], dtrain['target'], cv=cv_folds, scoring='roc_auc')
    
    #Print model report:
    print "\nModel Report"
    # print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions)
    gini_trn = ml.gini_normalized(dtrain['target'], dtrain_predprob)
    gini_tst = ml.gini_normalized(dtest['target'], dtest_predprob)
    print "gini score (Train): %f, (Test) %f" % (gini_trn, gini_tst)
    # print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['target'], dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp[:50].plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()




param_test1 = {'n_estimators':range(20,81,10)}

gsearch1 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		min_samples_split=500,
		min_samples_leaf=50,
		max_depth=8,
		max_features='sqrt',
		subsample=0.8,
		random_state=10), 
	param_grid = param_test1, 
	scoring=my_scorer,
	n_jobs=4,
	iid=False, 
	cv=5)


start_time = time.time()
gsearch1.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


n_estimators = 40



param_test2 = {'max_depth':range(5,16,2), 'min_samples_split':range(100,1001,200)}

gsearch2 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators, 
		max_features='sqrt', 
		subsample=0.8, 
		random_state=10),
	param_grid = param_test2, 
	scoring=my_scorer,
	n_jobs=4,
	iid=False, 
	cv=5 )


start_time = time.time()
gsearch2.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_



#==

param_test3 = {'max_depth':[3, 4, 5], 'min_samples_split':range(900,1301,200)}

gsearch3 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators, 
		max_features='sqrt', 
		subsample=0.8, 
		random_state=10),
	param_grid = param_test3, 
	scoring=my_scorer,
	n_jobs=4,
	iid=False, 
	cv=5 )


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_



max_depth = 3

param_test3 = {'min_samples_split':range(100,1301,200)}

gsearch3 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators, 
		max_depth=max_depth, 
		max_features='sqrt', 
		subsample=0.8, 
		random_state=10),
	param_grid = param_test3, 
	scoring=my_scorer,
	n_jobs=4,
	iid=False, 
	cv=5 )


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_


# == 

# param_test3 = {'min_samples_split':range(100,1301,200), 'min_samples_leaf':range(30,71,10)}
param_test3 = {'min_samples_split':range(1100,1501,100), 'min_samples_leaf':range(20,41,5)}

gsearch3 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators,
		max_depth=max_depth,
		max_features='sqrt', 
		subsample=0.8, 
		random_state=10),
	param_grid = param_test3, 
	scoring=my_scorer,
	n_jobs=-1,
	iid=False, 
	cv=5)


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_




modelfit(gsearch3.best_estimator_, df_train2, use_features)

modelfit(gsearch3.best_estimator_, df_train, use_features)


min_samples_leaf = 25 
min_samples_split = 1500

param_test4 = {'max_features':range(25, 801, 50)}
param_test4 = {'max_features':[600, 625, 650]}

gsearch4 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators,
		max_depth=max_depth, 
		min_samples_split=min_samples_split, 
		min_samples_leaf=min_samples_leaf, 
		subsample=0.8, 
		random_state=10),
	param_grid = param_test4, 
	scoring=my_scorer,
	n_jobs=-1, 
	iid=False, 
	cv=5 )

start_time = time.time()
gsearch4.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

max_features = 625

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}

gsearch5 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		learning_rate=0.1, 
		n_estimators=n_estimators,
		max_depth=max_depth,
		min_samples_split=min_samples_split, 
		min_samples_leaf=min_samples_leaf, 
		random_state=10,
		max_features=max_features ),
	param_grid = param_test5, 
	scoring=my_scorer,
	n_jobs=-1,
	iid=False, 
	cv=5 )

start_time = time.time()
gsearch5.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_




n_estimators = 40
max_depth = 3
min_samples_split = 1500
min_samples_leaf = 25
max_features = 625
subsample = 0.8


ml.modelfit(gsearch5.best_estimator_, df_train2, df_val2, use_features, printFeatureImportance=False)

gbm_tuned_1 = GradientBoostingClassifier(
	learning_rate=0.05, 
	n_estimators=n_estimators*2,
	max_depth=max_depth, 
	min_samples_split=min_samples_split,
	min_samples_leaf=min_samples_leaf, 
	subsample=subsample, 
	random_state=10, 
	max_features=max_features)

ml.modelfit(gbm_tuned_1, df_train2, df_val2, use_features, printFeatureImportance=False)


param_test6 = {'learning_rate':[0.01, 0.05, 0.1], 'n_estimators':[40, 80, 100]}

gsearch6 = GridSearchCV(
	estimator = GradientBoostingClassifier(
		max_depth=max_depth,
		min_samples_split=min_samples_split, 
		min_samples_leaf=min_samples_leaf, 
		random_state=10,
		max_features=max_features ),
	param_grid = param_test6, 
	scoring=my_scorer,
	n_jobs=-1,
	iid=False, 
	cv=5 )

start_time = time.time()
gsearch6.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

ml.modelfit(gsearch6.best_estimator_, df_train2, df_val2, use_features, printFeatureImportance=False)



gbm_tuned_2 = GradientBoostingClassifier(
	learning_rate=0.05, 
	n_estimators=200,
	max_depth=max_depth, 
	min_samples_split=min_samples_split,
	min_samples_leaf=min_samples_leaf, 
	subsample=subsample, 
	random_state=10, 
	max_features=max_features)

ml.modelfit(gbm_tuned_2, df_train, df_val, use_features, printFeatureImportance=False)


gbm_tuned_3 = GradientBoostingClassifier(
	learning_rate=0.01, 
	n_estimators=400,
	max_depth=max_depth, 
	min_samples_split=min_samples_split,
	min_samples_leaf=min_samples_leaf, 
	subsample=subsample, 
	random_state=10, 
	max_features=max_features)

ml.modelfit(gbm_tuned_3, df_train, df_val, use_features, printFeatureImportance=False)



gbm_tuned_4 = GradientBoostingClassifier(
	learning_rate=0.005, 
	n_estimators=300,
	max_depth=max_depth, 
	min_samples_split=min_samples_split,
	min_samples_leaf=min_samples_leaf, 
	subsample=subsample, 
	random_state=10, 
	max_features=max_features)

ml.modelfit(gbm_tuned_4, df_train, df_val, use_features, performCV=False, printFeatureImportance=False)










