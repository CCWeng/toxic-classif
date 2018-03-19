import pandas as pd
import numpy as np
import pickle as pkl

import operator
import itertools
import sys
import os
import re

from matplotlib import dates as dates
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import animation

from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import chi2

from sklearn.model_selection import TimeSeriesSplit

import scipy.stats as stats
from scipy.stats import entropy
from scipy.stats import norm


from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV

from pandas.tools.plotting import scatter_matrix

from datetime import datetime

import fasttext as ft

import itertools

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

target = 'toxic'


import unicodedata





def GridSearchFastText(f_trn, f_tst, params, silent=True):
	results = list()
	
	max_p = -float('inf')
	max_r = -float('inf')

	best_p_param = dict()
	best_r_param = dict()

	for n_gram, epoch, lr in itertools.product(params['word_ngrams'], params['epoch'], params['lr']):
		clf = ft.supervised(f_trn, 'ft_model', word_ngrams=n_gram, epoch=epoch, lr=lr, silent=True)
		res = clf.test(f_tst)

		p = res[1]
		r = res[2]

		results.append((n_gram, epoch, lr, p, r))
		if p > max_p:
			max_p = p
			best_p_param['word_ngrams'] = n_gram
			best_p_param['epoch'] = epoch
			best_p_param['lr'] = lr

		if r > max_r:
			max_r = r
			best_r_param['word_ngrams'] = n_gram
			best_r_param['epoch'] = epoch
			best_r_param['lr'] = lr

	print "\n== Precision Grids ==\n"
	if not silent:
		for n_gram, epoch, lr, p, r in results:
			print "word_ngrams = %d, epoch = %d, lr = %.2f, precision = %.5f" % (n_gram, epoch, lr, p)
	print "best params : word_ngrams = %d, epoch = %d, lr = %.2f, precision = %.5f" \
		% (best_p_param['word_ngrams'], best_p_param['epoch'], best_p_param['lr'], max_p)

	print "\n== Recall Grids ==\n"
	if not silent:
		for n_gram, epoch, lr, p, r in results:
			print "word_ngrams = %d, epoch = %d, lr = %.2f, recall = %.5f" % (n_gram, epoch, lr, r)
	print "best params : word_ngrams = %d, epoch = %d, lr = %.2f, recall = %.5f" \
		% (best_p_param['word_ngrams'], best_p_param['epoch'], best_p_param['lr'], max_r)

	return best_p_param, best_r_param











def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def FeatureSelectGreedy(df, model, in_columns, target, step=100):
	y = df[target]

	selector = RFECV(model, step=1, cv=3)

	keep_columns = list()
	N = len(in_columns)

	for i in range(0, N, step):
		j = min(i+step, N)
		print "\n--\nNumber of test features = %d(/%d)" % (j, N) 		

		X = df[keep_columns + in_columns[i:j]]

		start_time = timer(None)

		selector = selector.fit(X, y)
		
		timer(start_time)

		keep_columns = X.columns[selector.support_].tolist()
		score = selector.score(X, y)

		print "Number of keep features =", len(keep_columns)
		print "Score =", score

	return keep_columns






def GetBagOfWords(df, tfidf_columns):
	n_select = 150

	X = (df[tfidf_columns] > 0).astype(int)
	vocabulary = [c.split('_tfidf')[0] for c in tfidf_columns]
	df_chi2 = pd.DataFrame(index=vocabulary)
	# df_chi2 = pd.DataFrame(index=[c[:-6] for c in tfidf_columns])

	bow = set()
	for tgt in targets:
		y = df[tgt] 
		chi2_val, p_val = chi2(X, y)
		df_chi2[tgt] = chi2_val
		df_chi2[tgt+'_order'] = (-chi2_val).argsort().argsort()

		words = df_chi2[df_chi2[tgt+'_order'] < n_select].index.tolist()
		bow = bow.union(words)

	bow = list(bow)
	bow = [unicodedata.normalize('NFKD', w).encode('ascii', 'ignore') for w in bow]

	return bow, df_chi2





def true_positive_rate(label, predict):
	if (predict == 0).all():
		return 0
	tpr = float((label & predict).sum()) / predict.sum()
	return tpr

def false_positive_rate(label, predict):
	if (predict == 0).all():
		return 0
	fpr = float(((1-label) & predict).sum()) / predict.sum()
	return fpr






'''
===============================
PERMUTATION IMPORTANCE (PIMP)
===============================
'''



def create_feature_map(features):
	outfile = open('xgb.fmap', 'w')
	i = 0
	for feat in features:
		outfile.write('{0}\t{1}\tq\n'.format(i, feat))
		i = i + 1

	outfile.close()


def ComputePermutationImportance(df_trn, use_columns, clf, eval_func):
	n_splits = 3
	n_runs = 5

	data = df_trn[use_columns]
	target = df_trn[target]

	imp_df = pd.DataFrame(np.ones((len(use_columns), n_splits * n_runs)), index=use_columns)
	np.random.seed(9385610)
	idx = np.arange(len(target))
	for run in range(n_runs):
		# Shuffle target
		np.random.shuffle(idx)
		perm_target = target.iloc[idx]
		# Create a new split
		folds = TimeSeriesSplit(n_splits)
		# folds = StratifiedKFold(n_splits, shuffle=True, random_state=None)
		oof = np.empty(len(df_trn))

		for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
			msg = "\rCompute permutation importance - run %d, fold %d ...      " % (run+1, fold_+1)
			sys.stdout.write(msg)
			sys.stdout.flush()

			trn_dat, trn_tgt = data.iloc[trn_idx], perm_target.iloc[trn_idx]
			val_dat, val_tgt = data.iloc[val_idx], perm_target.iloc[val_idx]
			# Train classifier
			clf.fit(trn_dat, trn_tgt)
			# Keep feature importances for this fold and run
			fscore = clf.booster().get_score(importance_type='gain')
			fea = fscore.keys()
			imp = fscore.values()
			imp_df.loc[fea, n_splits * run + fold_] = imp
			# Update OOF for gini score display
			oof[val_idx] = clf.predict(val_dat)
		    
		sys.stdout.write("done.\n")
		print("Run %2d OOF score : %.6f" % (run, eval_func(perm_target, oof)))


	bench_imp_df = pd.DataFrame(np.ones((len(use_columns), n_splits * n_runs)), index=use_columns)
	idx = np.arange(len(target))
	n_choice = int(len(idx) * 0.8)
	for run in range(n_runs):
		# Shuffle target
		choice_idx = np.random.choice(idx, n_choice)
		perm_target = target.iloc[choice_idx]
		perm_data = data.iloc[choice_idx]
		    
		# Create a new split
		folds = TimeSeriesSplit(n_splits)
		oof = np.empty(len(df_trn))

		for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
			msg = "\rCompute bench importance - run %d, fold %d ...      " % (run+1, fold_+1)
			sys.stdout.write(msg)
			sys.stdout.flush()

			trn_dat, trn_tgt = data.iloc[trn_idx], target.iloc[trn_idx]
			val_dat, val_tgt = data.iloc[val_idx], target.iloc[val_idx]
			# Train classifier
			clf.fit(trn_dat, trn_tgt)
			# Keep feature importances for this fold and run
			fscore = clf.booster().get_score(importance_type='gain')
			fea = fscore.keys()
			imp = fscore.values()
			bench_imp_df.loc[fea, n_splits * run + fold_] = imp
			# Update OOF for gini score display
			oof[val_idx] = clf.predict(val_dat)

		sys.stdout.write('done.\n')
		print("Run %2d OOF score : %.6f" % (run, eval_func(perm_target, oof)))

	bench_mean = bench_imp_df.mean(axis=1)
	perm_mean = imp_df.mean(axis=1)

	pvalues = pd.concat([bench_mean, perm_mean], axis=1).reset_index()
	pvalues.columns=['feature', 'benchmark', 'permutation']
	pvalues['ratio'] = pvalues.benchmark / pvalues.permutation
	pvalues.sort_values(by='ratio', ascending=False, inplace=True)

	print("%-60s | benchmark | permutation | Ratio" % "Feature")
	for f, b, p, r in pvalues.values:
		print("%-60s |   %7.1f |     %7.1f |   %7.1f" % (f, b, p, r))

	return pvalues



'''
==================
FEATURE ANALYSIS
==================
'''




from sklearn.metrics import adjusted_mutual_info_score


def ComputeMutualInfo(df, columns):
	scores = list()	
	for c in columns:
		score = adjusted_mutual_info_score(df['target'], df[c])
		scores.append(score)

	mi = pd.Series(scores, columns)

	return mi












def ComputeHistSimilarity(df, columns):
	
	simi_list = list()
	for c in columns:
		values = np.sort(df[c].unique())
		counts = df.groupby('target')[c].value_counts()
		counts0 = counts[0]
		counts1 = counts[1]

		df_count = pd.DataFrame(0, index=values, columns=['0', '1'])
		df_count.loc[counts0.index, '0'] = counts0
		df_count.loc[counts1.index, '1'] = counts1

		vec0 = df_count['0'].values.astype(float) / df_count['0'].sum()
		vec1 = df_count['1'].values.astype(float) / df_count['1'].sum()

		norm0 = np.sqrt((vec0**2).sum())
		norm1 = np.sqrt((vec1**2).sum())

		similarity = (vec0.dot(vec1)) / (norm0 * norm1)
		simi_list.append(similarity)

	df_simi = pd.Series(simi_list, columns)

	return df_simi












## Two-sample Z statistics for grouping measurement
## OvA 
def Ztest(df, group_column, target_column):
	gsize_thresh = 20

	cate_values = df[group_column].unique()
	group_z_stats = list()
	group_size = list()


	num_samples = df.shape[0]
	num_grouping_samples = 0.
	for cv in cate_values:
		if str(cv) != 'nan':
			g1 = df.loc[df[group_column] == cv, target_column]
			g2 = df.loc[df[group_column] != cv, target_column]
		else:
			g1 = df.loc[df[group_column].isnull(), target_column]
			g2 = df.loc[~df[group_column].isnull(), target_column]

		n1 = g1.size
		n2 = g2.size

		if n1 < gsize_thresh or n2 < gsize_thresh:
			continue

		num_grouping_samples += n1

		mu1 = g1.mean()
		mu2 = g2.mean()

		sigma1 = g1.std()
		sigma2 = g2.std()

		delta = np.abs(mu1 - mu2)
		sigma_delta = np.sqrt(sigma1**2/n1 + sigma2**2/n2)
		
		z_stats = delta / sigma_delta

		group_z_stats.append(z_stats)
		group_size.append(n1)

	# print "The number of groups is %d (all = %d)." % (len(group_z_stats), len(cate_values))
	# print "The number of samples of all groups is %d (ratio = %f)." % (num_grouping_samples, num_grouping_samples / num_samples)

	group_z_stats = np.array(group_z_stats)
	group_size = np.array(group_size)

	global_z_stats = (group_z_stats * group_size).sum() / group_size.sum()

	return global_z_stats






## F-statistic for grouping measurement
def Ftest(df, group_column, target_column):
	grouped = df.groupby(group_column)[target_column]

	# Compute SSB (sum of square between)
	ssb = 0.
	total_mean = df[target_column].mean()
	for k, g in grouped:
		group_mean = g.mean()
		group_size = g.size
		ssb += ((group_mean - total_mean) ** 2) * group_size

	dfb = len(grouped) - 1


	# Compute SSW (sum of square within)
	ssw = 0.
	dfw = 0
	for k, g in grouped:
		group_samples = g.values
		group_mean = g.mean()
		for s in group_samples:
			ssw += (s - group_mean) ** 2
		dfw += (g.size - 1)

	# F-statistic
	f_statistic = (ssb / dfb) / (ssw / dfw)

	return f_statistic




def GroupMeasurement(df, measure_columns, group_column, measure_func=Ftest):
	measure_scores = list()
	for mc in measure_columns:
		score = measure_func(df, group_column, mc)
		measure_scores.append(score)

	feat_score = pd.Series(measure_scores, measure_columns).sort_values(ascending=False)
	feat_score[:100].plot(kind='bar')
	plt.ylabel('Grouping Score')
	plt.show()

	return feat_score











def FeatureAvblCluterCount(df):
	feature_count = df.shape[1]
	cluster_count = 2 ** feature_count
	
	counter = dict()
	for i in range(df.shape[0]):
		vec = df.iloc[i].copy()
		index1 = vec[~vec.isnull()].index
		index0 = vec[vec.isnull()].index
		vec[index0] = 0
		vec[index1] = 1

		bin_arr = vec.values.astype(int)
		int_val = BinArrToInt(bin_arr)
		cls = str(int_val)

		counter[cls] = counter.get(cls, 0) + 1

	return counter



def BinArrToInt(bin_arr):
	n_bits = len(bin_arr)
	val = 0
	for i in range(n_bits):
		val += (bin_arr[i] * (2**i))

	return val







'''
===============
PLOT ANALYSIS
===============
'''



def PlotScatterInColumns(df, columns):

	df0 = df.loc[df.target == 0, columns]
	df1 = df.loc[df.target == 1, columns]

	N = len(columns)
	for i in range(N):
		c1 = columns[i]
		for j in range(i+1, N):
			c2 = columns[j]


			fig = plt.figure()
			ax1 = fig.add_subplot(111)

			ax1.scatter(df0[c1], df0[c2], s=11, c='g', label='target = 0')
			ax1.scatter(df1[c1], df1[c2], s=10, c='r', label='target = 1')
			plt.legend(loc='best');

			# df[df.target == 0].plot.scatter(x=c1, y=c2, color='g')
			# df[df.target == 1].plot.scatter(x=c1, y=c2, color='r')

			plt.xlabel(c1)
			plt.ylabel(c2)

			plt.show()

	return





def PlotHistByTarget(df, col):
	df.loc[df.target == 0, col].plot('hist', bins=100, facecolor='g')
	df.loc[df.target == 1, col].plot('hist', bins=100, facecolor='r')

	plt.xlabel(col)
	plt.legend(['target=0', 'target=1'], loc='best')
	plt.show()

	return





def PlotGroupDensity(df, group_columns, target_columns=['target']):
	N = len(group_columns)
	for gc in group_columns:
		for tc in target_columns:
			grouped = df.groupby(gc)
			labels = []
			for (key, grp) in grouped:
				if len(grp) == 0: continue
				grp[tc].plot(kind='density')
				labels.append("%s=%s (#=%d)" % (gc, key, len(grp)))

			plt.xlim(df[tc].min(), df[tc].max())
			plt.legend(labels, loc='best')
			plt.title("%s" % (tc))
			plt.show()

	return




def PlotFeatureDensityOnErrorSign(df, columns):
	for c in columns:
		df.groupby('logerror_sign')[c].plot('density')
		plt.title(c)
		plt.show()

	return




def PlotScatterMatrix(df, columns):
	for c in columns:
		try:
			scatter_matrix(df[[c, 'logerror']], diagonal='kde')
			plt.show()
		except:
			print c, ": scatter matrix error."
	return




def PlotCorrelation(data_frame, columns, target=target):
	use_columns = columns + [target]
	data = data_frame[use_columns].astype(float)
	names = use_columns
	# names = [str(c) for c in data.columns]

	correlations = data.corr()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0, len(names),1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names, rotation='vertical')
	ax.set_yticklabels(names)
	plt.show()

	return







