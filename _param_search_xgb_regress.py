
'''
==========
Ref - 01
==========
'''

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

target = 'toxic'


from xgboost.sklearn import XGBRegressor

from sklearn import cross_validation, metrics   #Additional scklearn functions
# from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import make_scorer 
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import matplotlib.pylab as plt

# %matplotlib inline
from matplotlib.pylab import rcParams

import model_learning as ml

import time



rcParams['figure.figsize'] = 12, 4


my_scorer = make_scorer(ml.rmse, greater_is_better=False)


df_trn2 = df_trn2.sort_values('visit_date')
df_val2 = df_val2.sort_values('visit_date')

X_trn = df_trn2[use_columns]
y_trn = df_trn2[target]
X_val = df_val2[use_columns]
y_val = df_val2[target]


tscv = TimeSeriesSplit(n_splits=5)

#: Choose all predictors except target & IDcols

xgb1 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=211,
    # n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'reg:linear',
    nthread=4,
    scale_pos_weight=1,
    seed=27)


# pdb.run("ml.xgbfit(xgb1, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False, early_stopping_rounds=100)")
ml.xgbfit(xgb1, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False, early_stopping_rounds=100)





param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

param_test1 = {
 'max_depth':[2, 3, 4],
 'min_child_weight':[4, 5, 6, 7]
}

param_test1 = {
 'min_child_weight':[7, 8, 9, 11]
}


gsearch1 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate =0.1, 
        n_estimators=211, 
        max_depth=3,
        min_child_weight=7, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1, 
        seed=27), 
    param_grid = param_test1, 
    # scoring='neg_mean_squared_error',
    scoring=my_scorer,
    # n_jobs=-1,
    iid=False, 
    cv=tscv.split(X_trn) )


start_time = time.time()
gsearch1.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


param_test2 = {
 'max_depth':[3, 4],
 'min_child_weight':[1, 2, 3]
}

gsearch2 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate=0.1, 
        n_estimators=140, 
        max_depth=3,
        min_child_weight=1, 
        gamma=0,
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test2, 
    scoring=my_scorer,
    n_jobs=-1,
    iid=False, 
    cv=tscv.split(X_trn))


start_time = time.time()
gsearch2.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


max_depth = 3
min_child_weight = 9


param_test3 = {
 'gamma':[i/10.0 for i in range(0, 5)]
}

gsearch3 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate =0.1, 
        n_estimators=211, 
        max_depth=3,
        min_child_weight=9, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test3, 
    scoring=my_scorer,
    # n_jobs=-1,
    iid=False, 
    cv=tscv.split(X_trn))


start_time = time.time()
gsearch3.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time


gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_

gamma = 0


xgb2 = XGBRegressor(
    learning_rate=0.1,
    n_estimators=211,
    max_depth=3,
    min_child_weight=9,
    # max_depth=5,
    # min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'reg:linear',
    nthread=4,
    scale_pos_weight=1,
    seed=27)


ml.xgbfit(xgb2, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False)






xgb2.fit(df_trn.iloc[idx_trn][use_columns], df_trn.iloc[idx_trn][target], eval_metric=ml.rmse_xgb)
        
#: Predict training set:
pred_trn = xgb2.predict(df_trn.iloc[idx_trn][use_columns])
pred_tst = xgb2.predict(df_trn.iloc[idx_tst][use_columns])

rmse_trn = ml.rmse(df_trn.iloc[idx_trn][target], pred_trn)
rmse_tst = ml.rmse(df_trn.iloc[idx_tst][target], pred_tst)
    

print rmse_trn, rmse_tst

'''
Model Report
gini score (Train): 0.440322, (Test) 0.241820

'''




param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch4 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate =0.1, 
        n_estimators=211, 
        max_depth=3,
        min_child_weight=9, 
        gamma=0, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test4, 
    scoring=my_scorer,
    # n_jobs=-1,
    iid=False, 
    cv=tscv.split(X_trn))


start_time = time.time()
gsearch4.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time


gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_



param_test5 = {
 'subsample':[i/100.0 for i in range(85,100,5)],
 'colsample_bytree':[i/100.0 for i in range(50,70,5)]
}

gsearch5 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate =0.1, 
        n_estimators=59, 
        max_depth=3, 
        min_child_weight=1, 
        gamma=0.2, 
        subsample=0.8, 
        colsample_bytree=0.8,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test5, 
    scoring=my_scorer,
    # n_jobs=4,
    iid=False, 
    cv=tscv.split(X_trn))


start_time = time.time()
gsearch5.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time


gsearch5.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

subsample = 0.9
colsample_bytree = 0.7


param_test6 = {
 # 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
 'reg_alpha':[0.0001, 0.001, 0.01, 0.05, 0.1]
 # 'reg_alpha':[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
}

gsearch6 = GridSearchCV(
    estimator = XGBRegressor( 
        learning_rate =0.1, 
        n_estimators=211, 
        max_depth=3,
        min_child_weight=9, 
        gamma=0, 
        subsample=0.9, 
        colsample_bytree=0.7,
        objective= 'reg:linear', 
        nthread=4, 
        scale_pos_weight=1,
        seed=27), 
    param_grid = param_test6, 
    scoring=my_scorer,
    # n_jobs=4,
    iid=False, 
    cv=tscv.split(X_trn))

start_time = time.time()
gsearch6.fit(X_trn, y_trn)
elapsed_time = time.time() - start_time
print elapsed_time

gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_


reg_alpha = 0.01


xgb3 = XGBRegressor(
    learning_rate =0.1,
    n_estimators=211,
    max_depth=3,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    # subsample=0.9,
    # colsample_bytree=0.7,
    # reg_alpha=0.01,
    objective= 'reg:linear',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )


ml.xgbfit(xgb3, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False)

ml.xgbfit(xgb1, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False, early_stopping_rounds=100)

'''
Model Report
gini score (Train): 0.440315, (Test) 0.241823
'''



xgb4 = XGBRegressor(
    learning_rate =0.01,
    n_estimators=3214,
    # n_estimators=5000,
    max_depth=3,
    min_child_weight=9,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.7,
    reg_alpha=0.01,
    objective= 'reg:linear',
    nthread=4,
    scale_pos_weight=1,
    seed=27 )

ml.xgbfit(xgb4, df_trn2, df_val2, use_columns, useTrainCV=True, folds=tscv.split(X_trn), printFeatureImportance=False, early_stopping_rounds=200)
ml.xgbfit(xgb4, df_trn2, df_val2, use_columns, useTrainCV=False, printFeatureImportance=False)
ml.xgbfit(xgb4, df_trn, df_val, use_columns, useTrainCV=False, printFeatureImportance=False)

ml.xgbfit(xgb4, df_trn, df_val, pimp_features, useTrainCV=False, printFeatureImportance=False)







