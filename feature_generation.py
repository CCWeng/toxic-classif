import numpy as np
import pandas as pd
import sys

import data_process as dp
import feature_analysis as fa

from itertools import combinations


from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

from sklearn.linear_model import Ridge


seed = 7
num_folds = 5

target = 'visitors'



from datetime import date
from dateutil.relativedelta import relativedelta

import time



def GetTrend(df_trn, df_tst, features):
	
	model = Ridge()
	new_c = 'ridge_trend'

	n_folds = 5
	cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15) 

	df_trn.sort_values(by='visit_date', inplace=True)
	df_tst.sort_values(by='visit_date', inplace=True)

	for lrn_idx, prd_idx in cv.split(df_trn[features], df_trn[target]):
		lrn_I = df_trn.iloc[lrn_idx].index
		prd_I = df_trn.iloc[prd_idx].index

		model.fit(df_trn.loc[lrn_I, features], df_trn.loc[lrn_I, target])
		df_trn.loc[prd_I, new_c] = model.predict(df_trn.loc[prd_I, features])

	# Create feature values for test dataset
	model.fit(df_trn[features], df_trn[target])
	df_tst[new_c] = model.predict(df_tst[features])

	# fill na
	fval = df_trn[new_c].mean()
	df_trn[new_c] = df_trn[new_c].fillna(fval)
	df_tst[new_c] = df_tst[new_c].fillna(fval)

	return df_trn, df_tst


def CreateTrendColumns(df_trn, df_tst):
	trend_features = [
	'dom', 'doy', 'doq', 'dot', 'dow',
	'woq', 'wom', 'moq',
	'year', 'quarter', 'month', 'week',
	'dayofwork', 'dayofholiday' ]

	assert (df_trn[trend_features].isnull().sum().sum() == 0), 'Error : features contain NaN in train data.'
	assert (df_tst[trend_features].isnull().sum().sum() == 0), 'Error : features contain NaN in test data.'

	new_columns = ['ridge_trend']

	df_trn['ridge_trend'] = np.nan
	df_tst['ridge_trend'] = np.nan

	stores = list(set(df_trn.air_store_id.unique()).union(df_tst.air_store_id.unique()))
	N = len(stores)

	for i, s in enumerate(stores):
		msg = "\rCompute TREND features for each store (%.2f%%).      " % ((float(i+1) / N) * 100)
		sys.stdout.write(msg)
		sys.stdout.flush()

		trn = df_trn.loc[df_trn.air_store_id == s, trend_features+['visit_date', target]]
		tst = df_tst.loc[df_tst.air_store_id == s, trend_features+['visit_date', target]]

		if len(trn) < 150: continue

		trn, tst = GetTrend(trn, tst, trend_features)

		df_trn.loc[trn.index, new_columns] = trn[new_columns]
		df_tst.loc[tst.index, new_columns] = tst[new_columns]

	sys.stdout.write("\n")

	return df_trn, df_tst, new_columns



def CreateHolidayWeekCount(df_trn, df_tst):
	new_columns = ['holiday_cnt_currweek', 'holiday_cnt_lastweek', 'holiday_cnt_nextweek']

	df_trn['train'] = 1
	df_tst['train'] = 0
	df_trn['row_id'] = range(len(df_trn))
	df_tst['row_id'] = range(len(df_tst))

	use_columns = ['train', 'row_id', 'year', 'week', 'dow', 'holidaygen_flg']
	df_all = pd.concat([df_trn[use_columns], df_tst[use_columns]], axis=0, ignore_index=True)

	holiday_weektab = df_all.groupby(['year', 'week', 'dow'], as_index=False)['holidaygen_flg'].sum()
	holiday_weektab.loc[holiday_weektab.holidaygen_flg != 0, 'holidaygen_flg'] = 1

	holiday_weektab = holiday_weektab.groupby(['year', 'week'], as_index=False)['holidaygen_flg'].sum()
	holiday_weektab = holiday_weektab.rename(columns = {'holidaygen_flg': 'holiday_cnt_currweek'})
	holiday_weektab['holiday_cnt_lastweek'] = holiday_weektab.holiday_cnt_currweek.shift(1)
	holiday_weektab['holiday_cnt_nextweek'] = holiday_weektab.holiday_cnt_currweek.shift(-1)

	df_all = pd.merge(df_all, holiday_weektab, how='left', on=['year', 'week'])

	df_trn_new = df_all[df_all.train == 1].sort_values(by='row_id')
	df_tst_new = df_all[df_all.train == 0].sort_values(by='row_id')

	df_trn[new_columns] = df_trn_new[new_columns]
	df_tst[new_columns] = df_tst_new[new_columns]

	df_trn.drop(['train', 'row_id'], axis=1, inplace=True)
	df_tst.drop(['train', 'row_id'], axis=1, inplace=True)

	return df_trn, df_tst, new_columns




def GetFlagStats(df, types, ranges, flag):
	N = len(df)
	stats_dict = dict()
	for s in types:
		for r in ranges:
			stats_dict["%s_%s_%dmonth(s)" % (flag, s, r)] = np.full((N,), np.nan)

	for i in range(N):
		dt = df.iloc[i].visit_date
		flg = df.iloc[i][flag]
		rng_end = dt.replace(day=1) - relativedelta(days=1)

		tmp1 = (df.visit_date <= rng_end) & (df[flag] == flg)
		
		for rng in ranges:
			rng_begin = (dt - relativedelta(months=rng)).replace(day=1)

			tmp = (df.visit_date >= rng_begin) & tmp1
			rec = df.loc[tmp, 'visitors']
			if len(rec) == 0:
				continue

			assert (~np.isnan(rec)).all, "Error : NaN in visitors record."

			stats_dict["%s_cnt_%dmonth(s)" % (flag, rng)][i] = rec.count()
			stats_dict["%s_mean_%dmonth(s)" % (flag, rng)][i] = rec.mean()
			stats_dict["%s_med_%dmonth(s)" % (flag, rng)][i] = rec.median()
			stats_dict["%s_hmean_%dmonth(s)" % (flag, rng)][i] = hmean(rec.replace(0, 0.1))
			stats_dict["%s_std_%dmonth(s)" % (flag, rng)][i] = rec.std()
			stats_dict["%s_skew_%dmonth(s)" % (flag, rng)][i] = rec.skew()
			stats_dict["%s_kurt_%dmonth(s)" % (flag, rng)][i] = rec.kurt()
			stats_dict["%s_pct10_%dmonth(s)" % (flag, rng)][i] = rec.quantile(0.1)
			stats_dict["%s_pct90_%dmonth(s)" % (flag, rng)][i] = rec.quantile(0.9)

	return pd.DataFrame(stats_dict, index=df.index)


	


def CreateFlagHistory(df_trn, df_tst, flag):
	stats_ranges = [1, 2, 3, 6, 12] # months
	stats_types = ['cnt', 'mean', 'med', 'hmean', 'std', 'skew', 'kurt', 'pct10', 'pct90']

	df_trn['train'] = 1
	df_tst['train'] = 0
	df_trn['row_id'] = range(len(df_trn))
	df_tst['row_id'] = range(len(df_tst))

	use_columns = ['air_store_id', 'visit_date', flag, 'visitors', 'train', 'row_id']
	df_all = pd.concat([df_trn[use_columns], df_tst[use_columns]], axis=0, ignore_index=True)

	new_columns = list()
	for s in stats_types:
		for r in stats_ranges:
			c = "%s_%s_%dmonth(s)" % (flag, s, r)
			new_columns.append(c)
			df_all[c] = np.nan

	stores = list(df_all.groupby('air_store_id').size().sort_values(ascending=False).index)
	df_new = pd.DataFrame()

	N = len(stores)
	df_new = pd.DataFrame()
	for i, s in enumerate(stores):
		start_time = time.time()

		df_store = df_all.loc[df_all.air_store_id == s, ['visit_date', 'visitors', flag]]
		df_stats = GetFlagStats(df_store, stats_types, stats_ranges, flag)
		df_all.loc[df_stats.index, new_columns] = df_stats[new_columns]

		elapsed_time = time.time() - start_time
		n_record = len(df_store)

		msg = "\rComputing process %.2f%% (%d records, %.2fs)...      " % (float(i+1) / N * 100, n_record, elapsed_time)
		sys.stdout.write(msg)
		sys.stdout.flush()

	sys.stdout.write("done.\n")

	df_trn_new = df_all[df_all.train == 1].sort_values(by='row_id')
	df_tst_new = df_all[df_all.train == 0].sort_values(by='row_id')

	df_trn[new_columns] = df_trn_new[new_columns]
	df_tst[new_columns] = df_tst_new[new_columns]

	df_trn.drop(['row_id', 'train'], axis=1, inplace=True)
	df_tst.drop(['row_id', 'train'], axis=1, inplace=True)

	return df_trn, df_tst, new_columns






#===== HISTORY COLUMNS II

def my_extrapolate(series, degrade_ratio):
	old_index = series.index
	series.index = range(len(series))

	index = series[~series.isnull()].index
	
	end_idx1 = index.min()
	end_val1 = series[end_idx1]

	for i in range(end_idx1):
		series.iloc[i] = end_val1 * (degrade_ratio ** (end_idx1 - i))
	

	end_idx2 = index.max()
	end_val2 = series[end_idx2]	
	for i in range(end_idx2+1, len(series)):
		series.iloc[i] = end_val2 * (degrade_ratio ** (i - end_idx2))

	series.index = old_index

	return series

def my_interpolate(series, degrade_ratio):
	inter_index = series[series.isnull()].index
	series.interpolate(inplace=True)
	series[inter_index] *= degrade_ratio

	return series

def get_maxshift(series):
	idx = series[~series.isnull()].index.min()
	return len(series) - len(series[:idx]) - 1

def get_shiftsteps(total_shift, shift_size):
	shift_steps = list()
	rest_shift = total_shift
	while rest_shift > shift_size:
		shift_steps.append(shift_size)
		rest_shift -= shift_size

	shift_steps.append(rest_shift)
	return shift_steps


def compute_period_stats3(ts, period_length, timepoints, compute_std=True):
	assert period_length > 0, 'Error : period_length must be larger than 0.'

	rolling = ts.rolling(window=period_length, min_periods=1)
	weights = rolling.count().replace(0, np.nan) / float(period_length)

	means = rolling.mean()	
	if compute_std:
		stds = rolling.std()
	
	max_shift = get_maxshift(weights)

	df = pd.DataFrame(index=ts.index)
	for tp in timepoints:
		interval = (tp, tp + period_length - 1)
		shift_steps = get_shiftsteps(tp, max_shift)

		weights_shift = pd.Series(weights, copy=True)
		means_shift = pd.Series(means, copy=True)
		for ss in shift_steps:
			weights_shift = weights_shift.shift(ss)
			weights_shift = my_extrapolate(weights_shift, 0.95)
			weights_shift = my_interpolate(weights_shift, 0.95)
			means_shift = means_shift.shift(ss).interpolate(limit_direction='both')

		df['visitors_mean_%d-%d_in_past' % interval] = means_shift
		df['visitors_weighted_mean_%d-%d_in_past' % interval] = means_shift * weights_shift

		if compute_std:
			stds_shift = pd.Series(stds, copy=True)
			for ss in shift_steps:
				stds_shift = stds_shift.shift(ss).interpolate(limit_direction='both')

			df['visitors_std_%d-%d_in_past' % interval] = stds_shift
			df['visitors_weighted_std_%d-%d_in_past' % interval] = stds_shift * weights_shift

	return df



def CreateHistoryColumns3(df_trn, df_tst, use_tst_target=False):
	df_trn['train'] = 1
	df_tst['train'] = 0

	use_columns = ['air_store_id', 'visit_date', 'visitors', 'train']

	df_all = pd.concat([df_trn[use_columns], df_tst[use_columns]], axis=0, ignore_index=True)
	# df_all = df_trn[use_columns].append(df_tst[use_columns], ignore_index=True)

	if not use_tst_target:
		df_all.loc[df_all.train==0, 'visitors'] = np.nan

	stores = df_all.air_store_id.unique()
	n_stores = len(stores)
	for i, sid in enumerate(stores):
		msg = '\rCreate visitor history for stores (%.2f%%) ...       ' % ((float(i+1) / n_stores) * 100)
		sys.stdout.write(msg)
		sys.stdout.flush()

		# Gather record		
		visit_date = pd.to_datetime(df_all.loc[df_all.air_store_id == sid, 'visit_date'])
		visitors = df_all.loc[df_all.air_store_id == sid, 'visitors']

		sdate = visit_date.min()
		edate = visit_date.max()

		record = pd.Series(np.nan, index=pd.date_range(sdate, edate))
		record[visit_date] = visitors

		if np.all(record.isnull()): continue

		# Create history features
		df_1  = compute_period_stats3(record,  1, [i*7 for i in [1, 2, 3, 4, 5, 6, 7]], compute_std=False)
		df_7  = compute_period_stats3(record,  7, [i*7+1 for i in [0, 1, 2, 3, 4, 5, 6, 7]])
		df_30 = compute_period_stats3(record, 30, [i*30+1 for i in [0, 1, 2, 3]])
		df_60 = compute_period_stats3(record, 60, [i*60+1 for i in [0, 1]])
		df_90 = compute_period_stats3(record, 90, [1])

		df_new = pd.concat([df_1, df_7, df_30, df_60, df_90], axis=1)

		new_columns = ['store_' + c for c in df_new.columns]
		df_new.columns = new_columns

		if not set(new_columns).issubset(df_all.columns):
			df_all = pd.concat([df_all, pd.DataFrame(columns=new_columns, index=df_all.index)], axis=1)

		# Merge back created features
		index = df_all.loc[df_all.air_store_id==sid].index
		dates = pd.DatetimeIndex(df_all.loc[df_all.air_store_id==sid].visit_date)
		df_all.loc[index, new_columns] = df_new.loc[dates].values

	sys.stdout.write('done.\n')
	
	for c in new_columns:
		df_trn[c] = 0
		df_tst[c] = 0
		if df_all[c].isnull().sum():
			df_all[c].fillna(df_all[c].mean(), inplace=True)
		
	df_trn[new_columns] = df_all.loc[df_all.train==1, new_columns]
	df_tst[new_columns] = df_all.loc[df_all.train==0, new_columns]
	
	return df_trn, df_tst, new_columns





#===== HISTORY COLUMNS I 

def impute_visitors_by_weekday(ts):
	weekday = pd.Series(ts.index.dayofweek, index=ts.index)

	for wd in range(7):
		index = weekday[weekday==wd].index
		ts[index] = ts[index].interpolate(limit_direction='backward')

	return ts



def compute_period_stats(ts, period_length, timepoints, compute_std=True):
	assert period_length > 0, 'Error : period_length must be larger than 0.'

	rolling = ts.rolling(window=period_length, min_periods=1)
	period_cnt = rolling.count()
	# weights = rolling.count().replace(0, np.nan) / float(period_length)

	period_mean = rolling.mean()
	# period_mean = pd.rolling_mean(ts, window=period_length, min_periods=1)
	# period_mean.interpolate(inplace=True, limit_direction='both')

	if compute_std:
		period_std = rolling.std()
		# period_std = pd.rolling_std(ts, window=period_length, min_periods=1)
		# period_std.interpolate(inplace=True, limit_direction='both')

	df = pd.DataFrame(index=ts.index)
	for tp in timepoints:		
		mean_c = 'visitors_mean_over_day_%d-%d_in_past' % (tp, tp+period_length-1)
		df[mean_c] = period_mean.shift(tp)

		count_c = 'visitors_count_over_day_%d-%d_in_past' % (tp, tp+period_length-1)
		df[count_c] = period_cnt.shift(tp)

		if compute_std:
			std_c = 'visitors_std_over_day_%d-%d_in_past' % (tp, tp+period_length-1)
			df[std_c] = period_std.shift(tp)

	return df



def CreateHistoryColumns(df_trn, df_tst, use_tst_target=False):
	df_trn['train'] = 1
	df_tst['train'] = 0

	use_columns = ['air_store_id', 'visit_date', 'visitors', 'train']

	df_all = pd.concat([df_trn[use_columns], df_tst[use_columns]], axis=0, ignore_index=True)
	# df_all = df_trn[use_columns].append(df_tst[use_columns])


	if not use_tst_target:
		df_all.loc[df_all.train==0, 'visitors'] = np.nan

	stores = df_all.air_store_id.unique()
	n_stores = len(stores)
	offset = (df_tst.visit_date.max() - df_tst.visit_date.min()).days

	for i, sid in enumerate(stores):
		msg = '\rCreate visitor history for stores (%.2f%%) ...       ' % ((float(i+1) / n_stores) * 100)
		sys.stdout.write(msg)
		sys.stdout.flush()

		# Gather record		
		visit_date = pd.to_datetime(df_all.loc[df_all.air_store_id == sid, 'visit_date'])
		visitors = df_all.loc[df_all.air_store_id == sid, 'visitors']

		sdate = visit_date.min()
		edate = visit_date.max()

		record = pd.Series(np.nan, index=pd.date_range(sdate, edate))
		record[visit_date] = visitors

		if np.all(record.isnull()): 
			continue

		df_1  = compute_period_stats(record,  1, [(int(np.ceil(offset / 7.))+i)*7 for i in [0, 1, 2, 3]], compute_std=False)
		df_7  = compute_period_stats(record,  7, [offset + i*7+1 for i in [0, 1, 2, 3]])
		df_30 = compute_period_stats(record, 30, [offset + i*30+1 for i in [0, 1]])
		df_60 = compute_period_stats(record, 60, [offset + i*60+1 for i in [0, 1]])

		df_new = pd.concat([df_1, df_7, df_30, df_60], axis=1)

		new_columns = ['store_' + c for c in df_new.columns]
		df_new.columns = new_columns

		if not set(new_columns).issubset(df_all.columns):
			df_all = pd.concat([df_all, pd.DataFrame(columns=new_columns, index=df_all.index)], axis=1)

		# Merge back created features
		index = df_all.loc[df_all.air_store_id==sid].index
		dates = pd.DatetimeIndex(df_all.loc[df_all.air_store_id==sid].visit_date)
		df_all.loc[index, new_columns] = df_new.loc[dates].values

	sys.stdout.write('done.\n')
	
	for c in new_columns:
		df_trn[c] = 0
		df_tst[c] = 0

	df_trn[new_columns] = df_all.loc[df_all.train==1, new_columns]
	df_tst[new_columns] = df_all.loc[df_all.train==0, new_columns]
	
	return df_trn, df_tst, new_columns




from scipy.stats import hmean



def compute_period_stats2(ts, period_length, timepoints):
	assert period_length > 0, 'Error : period_length must be larger than 0.'
	assert (ts<=0).sum() == 0, 'Error : the visitor record must be >= 0 (for hmean).'

	rol = ts.rolling(window=period_length, min_periods=1)


	stats = dict()
	stats['cnt'] = rol.count()
	stats['mean'] = rol.mean()
	stats['med'] = rol.median()
	stats['hmean'] = rol.apply(lambda x: hmean(x[~np.isnan(x)]))
	stats['std'] = rol.std()
	stats['skew'] = rol.skew()
	stats['kurt'] = rol.kurt()
	stats['pct10'] = rol.quantile(0.1)
	stats['pct90'] = rol.quantile(0.9)

	# weights = rolling.count().replace(0, np.nan) / float(period_length)
	# period_mean.interpolate(inplace=True, limit_direction='both')
	# period_std.interpolate(inplace=True, limit_direction='both')

	df = pd.DataFrame(index=ts.index)
	for tp in timepoints:
		for name, sval in stats.items():	
			col = '%s_on_%d-%d(%d)' % (name, tp+period_length-1, tp, period_length)
			df[col] = sval.shift(tp)

	return df



def CreateHistoryColumns2(df_trn, df_tst, use_tst_target=False):
	df_trn['train'] = 1
	df_tst['train'] = 0

	use_columns = ['air_store_id', 'visit_date', 'visitors', 'train']

	df_all = pd.concat([df_trn[use_columns], df_tst[use_columns]], axis=0, ignore_index=True)
	# df_all = df_trn[use_columns].append(df_tst[use_columns])


	if not use_tst_target:
		df_all.loc[df_all.train==0, 'visitors'] = np.nan

	stores = df_all.air_store_id.unique()
	n_stores = len(stores)
	offset = (df_tst.visit_date.max() - df_tst.visit_date.min()).days

	for i, sid in enumerate(stores):
		msg = '\rCreate visitor history for stores (%.2f%%) ...       ' % ((float(i+1) / n_stores) * 100)
		sys.stdout.write(msg)
		sys.stdout.flush()

		# Gather record		
		visit_date = pd.to_datetime(df_all.loc[df_all.air_store_id == sid, 'visit_date'])
		visitors = df_all.loc[df_all.air_store_id == sid, 'visitors']

		sdate = visit_date.min()
		edate = visit_date.max()

		record = pd.Series(np.nan, index=pd.date_range(sdate, edate))
		record[visit_date] = visitors

		if np.all(record.isnull()): 
			continue

		df_30  = compute_period_stats2(record,  30, [offset])
		df_60  = compute_period_stats2(record,  60, [offset])
		df_90  = compute_period_stats2(record,  90, [offset])
		df_180 = compute_period_stats2(record, 180, [offset])
		df_360 = compute_period_stats2(record, 360, [offset])

		df_new = pd.concat([df_30, df_60, df_90, df_180, df_360], axis=1)

		# new_columns = ['store_' + c for c in df_new.columns]
		# df_new.columns = new_columns

		new_columns = [c for c in df_new.columns]
		if not set(new_columns).issubset(df_all.columns):
			df_all = pd.concat([df_all, pd.DataFrame(columns=new_columns, index=df_all.index)], axis=1)

		# Merge back created features
		index = df_all.loc[df_all.air_store_id==sid].index
		dates = pd.DatetimeIndex(df_all.loc[df_all.air_store_id==sid].visit_date)
		df_all.loc[index, new_columns] = df_new.loc[dates].values

	sys.stdout.write('done.\n')
	
	for c in new_columns:
		df_trn[c] = 0
		df_tst[c] = 0

	df_trn[new_columns] = df_all.loc[df_all.train==1, new_columns]
	df_tst[new_columns] = df_all.loc[df_all.train==0, new_columns]
	
	return df_trn, df_tst, new_columns



#=========



def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def smoothing_encode(trn_series=None,
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



def CreateSmoothingColumns(df_trn, df_tst, in_columns, tgt_c, min_samples_leaf=200, smoothing=10, noise_level=0):
	new_columns = list()
	
	N = len(in_columns)
	for i, old_c in enumerate(in_columns):
		sys.stdout.write("\r : out-of-fold average feature for %s (%d/%d)                  " % (old_c, i+1, N))
		sys.stdout.flush()

		new_c = "smooth_%s@%s" % (tgt_c, old_c)

		df_trn[new_c], df_tst[new_c] = smoothing_encode(
			trn_series=df_trn[old_c],
			tst_series=df_tst[old_c],
			target=df_trn[tgt_c],
			min_samples_leaf=min_samples_leaf,
			smoothing=smoothing,
			noise_level=noise_level)

		new_columns.append(new_c)

	sys.stdout.write('\n')

	return df_trn, df_tst, new_columns




def week_of_quarter(dt):
	year = dt.year
	month = dt.month
	week = dt.week
	quarter = dt.quarter
	
	q_month = 1 + (quarter-1) * 3
	q_date = date(year, q_month, 1)
	q_week = q_date.isocalendar()[1]

	if month == 1:
		week = week if week <= 5 else 1
		q_week = q_week if q_week <= 5 else 1

	return week - q_week + 1


def week_of_month(dt):
	year = dt.year
	month = dt.month
	week = dt.week
	
	m_date = date(year, month, 1)
	m_week = m_date.isocalendar()[1]

	if month == 1:
		week = week if week <= 5 else 1
		m_week = m_week if m_week <= 5 else 1

	return week - m_week + 1


def month_of_quarter(df):
	year = df.year
	month = df.month
	quarter = df.quarter

	month_base = 1 + (quarter-1) * 3
	moq = month - month_base + 1

	return moq


def CreateTimestampColumns(date_series):
	date = pd.to_datetime(date_series)

	df = pd.DataFrame()
	df['year'] = date.dt.year
	df['quarter'] = date.dt.quarter
	df['month'] = date.dt.month
	
	df['week'] = date.dt.week
	df.loc[(df.month == 1) & (df.week > 5), 'week'] = 1

	df['wom'] = date.apply(week_of_month)
	df['woq'] = date.apply(week_of_quarter)
	df['moq'] = month_of_quarter(df)
	
	df['dom'] = date.dt.day
	df['doy'] = date.dt.dayofyear
	df['doq'] = day_of_quarter(date_series)
	df['dow'] = date.dt.dayofweek
	df['dot'] = (date.dt.date - date.dt.date.min()).apply(lambda x: x.days)

	return df



def day_of_quarter(date_series):
	date = pd.to_datetime(date_series)
	year = date.dt.year
	quarter = date.dt.quarter

	quarter_d1 = pd.Series(['%04d-%02d-01' % (y, 1+(q-1)*3) for y, q in zip(year, quarter)])
	quarter_d1 = pd.to_datetime(quarter_d1)

	days_in_quarter = (date - quarter_d1).apply(lambda x: x.days)
	days_in_quarter += 1

	return days_in_quarter



def CreateDateRelatedColumns(df_trn, df_tst):
	time_related_columns = list()

	# days in quarter
	df_trn['visit_dayofquarter'] = day_of_quarter(df_trn['visit_date'])
	df_tst['visit_dayofquarter'] = day_of_quarter(df_tst['visit_date'])
	time_related_columns.append('visit_dayofquarter')



	return df_trn, df_tst, time_related_columns


def CreatePeriodDayColumns(df):
	visit_datetime = pd.to_datetime(df.visit_date)

	df['visit_dayofyear'] = visit_datetime.dt.dayofyear
	df['quarter'] = visit_datetime.dt.quarter

	index = df.loc[df.quarter == 1].loc()




def FeatureGenerate(df_train, df_test, cate_features, nume_features):

	df_train, df_test, dumm_features = CreateDummyColumns(df_train, df_test, cate_features)

	df_train, df_test = dp.MissingValueImputation(df_train, df_test, nume_features)

	df_train, df_test, inv_features = CreateInverseFeatures(df_train, df_test, nume_features)

	df_train, df_test, poly_features = CreatePolynomialFeatures(df_train, df_test, nume_features)

	new_features = dumm_features + inv_features + poly_features

	return df_train, df_test, new_features



def FeatureGenerate2(df_train, df_test, cate_features, nume_features, new_features):
	df_train, df_test, magic_features = CreateMagicFeatures(df_train, df_test)

	orig_cate_features = list()
	for c in cate_features:
		if c.endswith('_cat'):
			orig_cate_features.append(c)

	df_train, df_test, cate2d_features = CreateInteractCateFeatures(df_train, df_test, orig_cate_features)

	df_mi = fa.ComputeMutualInfo(df_train, cate2d_features)
	cate2d_features = df_mi.sort_values(ascending=False).head(20).index.tolist()

	df_train, df_test, dumm2d_features = CreateDummyColumns(df_train, df_test, cate2d_features)


	# inv_features = list()
	# for c in new_features:
	# 	if c.endswith('_inv'):
	# 		inv_features.append(c)
			
	# df_train, df_test, inv_stats_features_2d = CreateStatsFeatures(df_train, df_test, inv_features, cate2d_features)

	# df_train.drop(cate2d_features, axis=1, inplace=True)
	# df_test.drop(cate2d_features, axis=1, inplace=True)

	new_features = magic_features + dumm2d_features
	# new_features = magic_features + dumm2d_features + inv_stats_features_2d

	return df_train, df_test, new_features




def RemoveUnivalueFeatures(df_train, df_test, use_features):
	univalue_features = list()
	for f in use_features:
		if df_train[f].value_counts(dropna=False).shape[0] == 1:
			univalue_features.append(f)

	n = len(univalue_features)
	if n > 0:
		print "There are %d uni-value features. Remove ... " % (n)
		df_train.drop(univalue_features, axis=1, inplace=True)
		df_test.drop(univalue_features, axis=1, inplace=True)
		for f in univalue_features:
			use_features.remove(f)

	return df_train, df_test, use_features




def CreateMagicFeatures(df_train, df_test):
	df_train['ps_car_13_magic'] = np.round(df_train['ps_car_13']**2*48400, 2)
	df_test['ps_car_13_magic'] = np.round(df_test['ps_car_13']**2*48400, 2)

	df_train['ps_car_15_magic'] = df_train['ps_car_15'] ** 2
	df_test['ps_car_15_magic'] = df_test['ps_car_15'] ** 2

	df_train['ps_car_12_magic'] = np.round(df_train['ps_car_12']**2, 4) * 10000
	df_test['ps_car_12_magic'] = np.round(df_test['ps_car_12']**2, 4) * 10000

	magic_features = ['ps_car_13_magic', 'ps_car_15_magic', 'ps_car_12_magic']

	return df_train, df_test, magic_features







def ConvertNumeFeatureToCate(df_train, df_test, in_columns):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape
	
	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = in_columns + ['train']
	df_all = df_train[use_columns].append(df_test[use_columns])

	new_columns = list()
	for c in in_columns:
		new_c = c + '_cat'
		df_all[new_c] = pd.Categorical(df_all[c]).codes
		new_columns.append(new_c)

	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	overlap_columns_tr = list(set(new_columns).intersection(df_train.columns))
	overlap_columns_te = list(set(new_columns).intersection(df_test.columns))

	if len(overlap_columns_tr) > 0:
		df_train.drop(overlap_columns_tr, axis=1, inplace=True)

	if len(overlap_columns_te) > 0:
		df_test.drop(overlap_columns_te, axis=1, inplace=True)


	df_train = df_train.join(df_train_new[new_columns])
	df_test = df_test.join(df_test_new[new_columns])

	df_train.drop('train', axis=1, inplace=True)
	df_test.drop('train', axis=1, inplace=True)

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(new_columns)
	
	return df_train, df_test, new_columns




def CreateInteractCateFeatures(df_train, df_test, in_columns):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape
	
	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = in_columns + ['train']
	df_all = df_train[use_columns].append(df_test[use_columns])

	new_columns = list()
	for c1, c2 in combinations(in_columns, 2):
		c = c1 + '-' + c2
		data = df_all[c1].astype(str) + '-' + df_all[c2].astype(str)
		data = pd.Categorical(data).codes
		df_all[c] = data

		new_columns.append(c)

	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	overlap_columns_tr = list(set(new_columns).intersection(df_train.columns))
	overlap_columns_te = list(set(new_columns).intersection(df_test.columns))

	if len(overlap_columns_tr) > 0:
		df_train.drop(overlap_columns_tr, axis=1, inplace=True)

	if len(overlap_columns_te) > 0:
		df_test.drop(overlap_columns_te, axis=1, inplace=True)


	df_train = df_train.join(df_train_new[new_columns])
	df_test = df_test.join(df_test_new[new_columns])

	df_train.drop('train', axis=1, inplace=True)
	df_test.drop('train', axis=1, inplace=True)

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(new_columns)
	
	return df_train, df_test, new_columns





def CreateLog1pFeatures(df_train, df_test, nume_columns):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape

	transformer = FunctionTransformer(np.log1p)

	log1p_columns = [c+'_log1p' for c in nume_columns]

	overlap_tr = list(set(df_train.columns).intersection(log1p_columns))
	if len(overlap_tr) > 0:
		df_train.drop(overlap_tr, axis=1, inplace=True)

	overlap_te = list(set(df_test.columns).intersection(log1p_columns))
	if len(overlap_te) > 0:
		df_test.drop(overlap_te, axis=1, inplace=True)

	df_log1p_tr = pd.DataFrame(transformer.transform(df_train[nume_columns]), columns=log1p_columns, index=df_train.index)
	df_log1p_te = pd.DataFrame(transformer.transform(df_test[nume_columns]), columns=log1p_columns, index=df_test.index)

	df_train = df_train.join(df_log1p_tr)
	df_test = df_test.join(df_log1p_te)

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(log1p_columns)
	
	return df_train, df_test, log1p_columns



 

def CreateInverseFeatures(df_train, df_test, nume_columns):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape

	inv_columns = [c + '_inv' for c in nume_columns]

	df_train[inv_columns] = 1 / df_train[nume_columns]
	df_test[inv_columns] = 1 / df_test[nume_columns]


	df_train, df_test, inf_columns = ImputeInfFeatures(df_train, df_test, inv_columns)
	inv_columns += inf_columns

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(inv_columns)
	
	return df_train, df_test, inv_columns



def ImputeInfFeatures(df_train, df_test, in_columns, verbose=False):
	
	if verbose:
		sys.stdout.write("Impute Inf data ... ")
		sys.stdout.flush

	inf_columns = list()

	for c in in_columns:
		inf_trn = df_train[c].apply(np.isinf)
		inf_tst = df_test[c].apply(np.isinf)

		if inf_trn.sum():
			new_c = c + '_inf'
			df_train[new_c] = 0
			df_test[new_c] = 0

			index = df_train[inf_trn & (df_train[c] > 0)].index
			df_train.loc[index, c] = 0
			df_train.loc[index, new_c] = 1

			index = df_train[inf_trn & (df_train[c] < 0)].index
			df_train.loc[index, c] = 0
			df_train.loc[index, new_c] = -1

			index = df_test[inf_tst & (df_test[c] > 0)].index
			df_test.loc[index, c] = 0
			df_test.loc[index, new_c] = 1

			index = df_test[inf_tst & (df_test[c] < 0)].index
			df_test.loc[index, c] = 0
			df_test.loc[index, new_c] = -1

			inf_columns.append(new_c)

		elif inf_tst.sum():
			df_test.loc[inf_tst, c] = df_train[c].mean()

	if verbose:
		sys.stdout.write("%d done.\n" % (len(inf_columns)))

	return df_train, df_test, inf_columns


def CreatePolynomialFeatures(df_train, df_test, nume_columns, interaction_only=True):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape

	N = len(nume_columns)

	poly = PolynomialFeatures(2, interaction_only=interaction_only)
	
	poly_mat_tr = poly.fit_transform(df_train[nume_columns])[:, N+1:]
	poly_fea_tr = poly.get_feature_names(nume_columns)[N+1:]

	poly_mat_te = poly.fit_transform(df_test[nume_columns])[:, N+1:]
	poly_fea_te = poly.get_feature_names(nume_columns)[N+1:]

	assert(poly_fea_tr == poly_fea_te)

	df_poly_tr = pd.DataFrame(data = poly_mat_tr, columns = poly_fea_tr, index = df_train.index)
	df_poly_te = pd.DataFrame(data = poly_mat_te, columns = poly_fea_te, index = df_test.index)

	overlap_tr = list(set(df_train.columns).intersection(poly_fea_tr))
	if len(overlap_tr) > 0:
		df_train.drop(overlap_tr, axis=1, inplace=True)

	overlap_te = list(set(df_test.columns).intersection(poly_fea_te))
	if len(overlap_te) > 0:
		df_test.drop(overlap_te, axis=1, inplace=True)

	df_train = df_train.join(df_poly_tr)
	df_test = df_test.join(df_poly_te)

	poly_features = poly_fea_tr

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(poly_features)
	
	return df_train, df_test, poly_features






def CreateOccuranceRatioFeatures(df_train, df_test, cate_columns):
	print "Shape before generation : train =", df_train.shape, ", test =", df_test.shape

	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = cate_columns + ['train']
	df_all = df_train[use_columns].append(df_test[use_columns])
	# df_all[cate_columns] == df_all[cate_columns].astype(str)
	
	num_samples = df_all.shape[0]
	new_columns = list()
	for c in cate_columns:
		se_count = df_all.groupby(c).size()
		se_ratio = se_count / num_samples

		new_c = c + '_occur_r'
		df_all[new_c] = df_all[c].map(se_ratio)

		nan_count = df_all[c].isnull().sum()
		if nan_count > 0:
			nan_ratio = float(nan_count) / num_samples
			df_all.loc[df_all[c].isnull(), new_c] = nan_ratio
			
		new_columns.append(new_c)


	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	overlap_columns_tr = list(set(new_columns).intersection(df_train.columns))
	overlap_columns_te = list(set(new_columns).intersection(df_test.columns))

	# assert(overlap_columns_tr == overlap_columns_te)

	if len(overlap_columns_tr) > 0:
		df_train.drop(overlap_columns_tr, axis=1, inplace=True)

	if len(overlap_columns_te) > 0:
		df_test.drop(overlap_columns_te, axis=1, inplace=True)


	df_train = df_train.join(df_train_new[new_columns])
	df_test = df_test.join(df_test_new[new_columns])

	df_train.drop('train', axis=1, inplace=True)
	df_test.drop('train', axis=1, inplace=True)

	print "Shape after generation : train =", df_train.shape, ", test =", df_test.shape
	print "# of generated columns =", len(new_columns)
	

	return df_train, df_test, new_columns




def add_laplace_noise(series, scale):
	return series + np.random.laplace(0, scale, len(series))


## create out-of-fold AVERAGE predictors
def CreateOOFColumns(df_train, df_test, categorical_columns, cv=None, tgt_c=target):
	# Create feature value for train dataset

	if cv is None:
		n_folds = 5
		cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15) 

	N = len(categorical_columns)
	new_columns = list()

	df_trn_new = pd.DataFrame(index=df_train.index)
	df_tst_new = pd.DataFrame(index=df_test.index)

	for i, cate_c in enumerate(categorical_columns):
		sys.stdout.write("\r : out-of-fold average feature for %s (%d/%d)     " % (cate_c, i+1, N))
		sys.stdout.flush()

		df_use = df_train[[cate_c, tgt_c]]

		new_c = "oof_%s@%s" % (tgt_c, cate_c)
		df_trn_new[new_c] = np.nan
		
		for lrn_idx, prd_idx in cv.split(df_use[cate_c], df_use[tgt_c]):
			lrn_idx = df_use.index[lrn_idx]
			prd_idx = df_use.index[prd_idx]

			target_mean = df_use.loc[lrn_idx].groupby(cate_c)[tgt_c].mean()
			df_trn_new.loc[prd_idx, new_c] = df_use.loc[prd_idx, cate_c].map(target_mean)
			
			if df_trn_new.loc[prd_idx, new_c].isnull().sum():
				fval = df_use.loc[lrn_idx, tgt_c].mean()
				df_trn_new.loc[prd_idx, new_c] = df_trn_new.loc[prd_idx, new_c].fillna(fval)

		# Create feature values for test dataset
		# df_tst_new[new_c] = np.nan
		target_mean = df_train.groupby(cate_c)[tgt_c].mean()
		df_tst_new[new_c] = df_test[cate_c].map(target_mean)
		if df_tst_new[new_c].isnull().sum():
			fval = df_train[tgt_c].mean()
			df_tst_new[new_c].fillna(fval, inplace=True)

		new_columns.append(new_c)

	del_columns = [c for c in df_train.columns if c in new_columns]
	df_train.drop(del_columns, axis=1, inplace=True)

	del_columns = [c for c in df_test.columns if c in new_columns]
	df_test.drop(del_columns, axis=1, inplace=True)

	df_train = pd.concat([df_train, df_trn_new], axis=1)
	df_test = pd.concat([df_test, df_tst_new], axis=1)

	sys.stdout.write('\n')


	return df_train, df_test, new_columns


	

	




def ExtractMajorMinorColumns(df, columns):
	major_columns = list()
	minor_columns = list()

	nan_thresh = 0.6
	n_samples = float(df.shape[0])
	for c in columns:
		nan_r = df[c].isnull().sum() / n_samples
		if nan_r < nan_thresh:
			major_columns.append(c)
		else:
			minor_columns.append(c)

	return major_columns, minor_columns



def RemoveColumns(columns, remove_columns):
	for c in remove_columns:
		if c in columns:
			columns.remove(c)

	return


def CreateDummyColumns(df_train, df_test, columns):

	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = ['train'] + columns
	df_all = df_train[use_columns].append(df_test[use_columns])

	df_dumm = pd.get_dummies(df_all[columns], dummy_na=True, columns=columns)
	dumm_columns = df_dumm.columns.tolist()

	df_all = pd.concat([df_all, df_dumm], axis=1)

	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	overlap_columns_tr = list(set(dumm_columns).intersection(set(df_train.columns)))
	overlap_columns_te = list(set(dumm_columns).intersection(set(df_test.columns)))
	
	assert(overlap_columns_tr == overlap_columns_te)
	
	if len(overlap_columns_tr) > 0:
		df_train.drop(overlap_columns_tr, axis=1, inplace=True)
		df_test.drop(overlap_columns_te, axis=1, inplace=True)

	df_train = df_train.join(df_train_new[dumm_columns])
	df_test = df_test.join(df_test_new[dumm_columns])

	df_train.drop('train', axis=1, inplace=True)
	df_test.drop('train', axis=1, inplace=True)

	return df_train, df_test, dumm_columns



def CateColumnsSplitByCardinality(df, columns):
	card_thresh = 200
	hi_card_columns = list()
	lo_card_columns = list()
	for c in columns:
		card = len(df[c].unique())
		if card > card_thresh:
			hi_card_columns.append(c)
		else:
			lo_card_columns.append(c)

	return hi_card_columns, lo_card_columns



def CreateLikelihoodEncodingColumns(df_train, df_test, columns):
	n_folds = 5
	n_samples = df_train.shape[0]
	fold_size = int((float(n_samples) / n_folds))

	fold_flags = np.zeros(n_samples)
	for i in range(n_folds):
		s = i * fold_size
		e = (i+1) * fold_size if i < (n_folds-1) else n_samples
		f = i + 1
		fold_flags[s:e] = f

	df_train['fold_flags'] = fold_flags

	folds = range(1, n_folds+1)

	new_columns = list()
	n_columns = len(columns)
	for i, c in enumerate(columns):
		msg = "Create likelihood encoding column for %s(%d/%d) ... " % (c, i+1, n_columns)
		sys.stdout.write(msg)
		sys.stdout.flush()

		new_c = c + '_tar_avg'
		new_columns.append(new_c)

		df_train[new_c] = 0
		df_test[new_c] = 0

		# Create train features using cross-validation
		for f in folds:
			tr_index = df_train[df_train['fold_flags'] != f].index
			te_index = df_train[df_train['fold_flags'] == f].index

			stats_table = TrainTargetStats(df_train.loc[tr_index], c)
			df_train.loc[te_index, new_c] = PredictTargetStats(df_train.loc[te_index], c, new_c, stats_table)

		# Create test features
		stats_table = TrainTargetStats(df_train, c)
		df_test[new_c] = PredictTargetStats(df_test, c, new_c, stats_table)

		sys.stdout.write("done.\n")


	df_train.drop('fold_flags', axis=1, inplace=True)

	return df_train, df_test, new_columns



def TrainTargetStats(df, col):
	stats_table = dict()
	values = df[col].unique()
	for v in values:
		if str(v) == 'nan':
			stats_table[str(v)] = df.loc[df[col].isnull(), 'logerror'].mean()
		else:
			stats_table[v] = df.loc[df[col]==v, 'logerror'].mean()

	return stats_table


def PredictTargetStats(df, col, new_col, stats_table):
	for old_v, new_v in stats_table.items():
		if old_v == 'nan':
			df.loc[df[col].isnull(), new_col] = new_v
		else:
			df.loc[df[col] == old_v, new_col] = new_v

	return df[new_col]






def CreateBinaryEncodingColumns(df_train, df_test, columns):
	df_train['row_id'] = range(df_train.shape[0])
	df_test['row_id'] = range(df_test.shape[0])
	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = ['row_id', 'train'] + columns

	df_all = df_train[use_columns].append(df_test[use_columns])
	df_all[columns] = df_all[columns].astype(str)

	new_columns = list()
	for c in columns:
		msg = "Create binary encoding columns for %s" % (c)
		sys.stdout.write(msg)
		sys.stdout.flush()
		cardinalities = df_all[c].unique().tolist()
		# print "column = %s, #cardinalities = %d" % (c, len(cardinalities))

		int_values = [cardinalities.index(v) for v in df_all[c]]
		bin_matrix = IntArr2BinMat(int_values)

		n_new = bin_matrix.shape[1]
		bin_columns = [c+"_b%d" % (i) for i in range(n_new)]

		df_all[bin_columns] = pd.DataFrame(bin_matrix, columns=bin_columns, index=df_all.index)		
		new_columns += bin_columns
		sys.stdout.write("done.\n")

	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	df_train_new.sort_values(by='row_id', inplace=True)
	df_test_new.sort_values(by='row_id', inplace=True)

	df_train[new_columns] = df_train_new[new_columns]
	df_test[new_columns] = df_test_new[new_columns]

	df_train.drop(['row_id', 'train'], axis=1, inplace=True)
	df_test.drop(['row_id', 'train'], axis=1, inplace=True)

	return df_train, df_test, new_columns



def IntArr2BinMat(in_arr):
	num_samples = len(in_arr)
	int_arr = np.reshape(np.array(in_arr), (num_samples, 1))
	max_value = int_arr.max()

	num_bits = len("{0:b}".format(max_value))

	bin_mat = np.array([])
	for b in range(num_bits-1, -1, -1):
		base = 2**b
		bin_arr = (int_arr >= base).astype(int)
		bin_mat = np.hstack((bin_mat, bin_arr)) if bin_mat.size else bin_arr
		int_arr -= (bin_arr * base)

	return bin_mat






	
def CreateSignProbColumns(df_train, df_test, columns, model):
	kfold = KFold(n_splits=num_folds, random_state=seed)

	sys.stdout.write("Extract data ... ")
	sys.stdout.flush()
	X_train = df_train[columns].astype('float')
	y_train = df_train['logerror_sign'].astype('int')
	X_val = df_test[columns].astype('float')
	sys.stdout.write("done.\n")

	sys.stdout.write("Create features for train data ... ")
	sys.stdout.flush()
	f_train = cross_val_predict(model, X_train, y_train, cv=kfold, method='predict_proba')
	f_train = f_train[:, 1] * 2 - 1
	sys.stdout.write("done.\n")

	sys.stdout.write("Create features for test data ... ")
	sys.stdout.flush()
	model.fit(X_train, y_train)
	f_test = model.predict_proba(X_val)
	f_test = f_test[:, 1] * 2 - 1
	sys.stdout.write("done.\n")

	new_col = 'logerror_sign_prob'
	# return f_train, f_test

	df_train[new_col] = f_train
	df_test[new_col] = f_test

	return df_train, df_test, [new_col]




def CreateNanRatioColumns(df_train, df_test, in_columns):
	n_samples = df_train.shape[0]
	nan_r_columns = list()
	
	for c in in_columns:
		n_nan = (df_train[c].isnull()).sum()
		r_nan = float(n_nan) / n_samples

		new_c = c + "_nan_r"
		df_train[new_c] = 0
		df_train.loc[df_train[c].isnull(), new_c] = r_nan
		df_train.loc[~df_train[c].isnull(), new_c] = (1 - r_nan)

		df_test[new_c] = 0
		df_test.loc[df_test[c].isnull(), new_c] = r_nan
		df_test.loc[~df_test[c].isnull(), new_c] = (1 - r_nan)

		nan_r_columns.append(new_c)

	return df_train, df_test, nan_r_columns



## FNSA : Feature's NaN State Accumulation
def CreateFnsaColumns(df_train, df_test, in_columns):
	df_train['row_id'] = range(df_train.shape[0])
	df_test['row_id'] = range(df_test.shape[0])
	
	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = ['row_id', 'train', 'transactiondate'] + in_columns
	df_all = df_train[use_columns].append(df_test[use_columns])
	df_all.sort_values(by='transactiondate', inplace=True)

	new_columns = list()
	for c in in_columns:
		new_c = c + '_nan_cond_accu'
		df_all[new_c] = 1

		index = df_all[df_all[c].isnull()].index
		df_all.loc[index, new_c] = df_all.loc[index, new_c].cumsum()

		index = df_all[~df_all[c].isnull()].index
		df_all.loc[index, new_c] = df_all.loc[index, new_c].cumsum()

		new_columns.append(new_c)


	new_train = df_all[df_all['train'] == 1]
	new_test = df_all[df_all['train'] == 0]
	new_train.sort_values('row_id', inplace=True)
	new_test.sort_values('row_id', inplace=True)

	df_train[new_columns] = new_train[new_columns]
	df_test[new_columns] = new_test[columns]

	df_train.drop(['row_id', 'train'], axis=1, inplace=True)
	df_test.drop(['row_id', 'train'], axis=1, inplace=True)

	return df_train, df_test, new_columns




def CreateStatsRatioFeatures(df_train, df_test, observe_columns, group_columns, replace=True):

	n_grp = len(group_columns)
	n_obs = len(observe_columns)

	new_columns = list()

	for i, gc in enumerate(group_columns):

		drop_columns = list()

		for j, oc in enumerate(observe_columns):

			mean_c = '%s_mean_on_%s' % (oc, gc)
			std_c = '%s_std_on_%s' % (oc, gc)
			med_col = '%s_median_on_%s' % (oc, gc)
			min_col = '%s_min_on_%s' % (oc, gc)
			max_col = '%s_max_on_%s' % (oc, gc)

			stats_columns = [mean_c, std_c, med_col, min_col, max_col]
			if np.array([c not in df_train.columns for c in stats_columns]).any():
				continue

			if np.array([c not in df_test.columns for c in stats_columns]).any():
				continue

			msg = "\rCreate statistics ratio columns of %s (%d/%d) on %s (%d/%d)           " % (oc, j+1, n_obs, gc, i+1, n_grp)
			sys.stdout.write(msg)
			sys.stdout.flush()

			meanrat_col = "%s_meanrat_on_%s" % (oc, gc)
			df_train[meanrat_col] = df_train[oc] / df_train[mean_c]
			df_test[meanrat_col] = df_test[oc] / df_test[mean_c]
			new_columns.append(meanrat_col)

			stdrat_col = "%s_stdrat_on_%s" % (oc, gc)
			df_train[stdrat_col] = (df_train[oc] - df_train[mean_c]) / df_train[std_c]
			df_test[stdrat_col] = (df_test[oc] - df_test[mean_c]) / df_test[std_c]
			new_columns.append(stdrat_col)
			
			range_col = "%s_range_on_%s" % (oc, gc)
			df_train[range_col] = df_train[max_col] - df_train[min_col]
			df_test[range_col] = df_test[max_col] - df_test[min_col]

			medrat_col = "%s_medrat_on_%s" % (oc, gc)
			df_train[medrat_col] = (df_train[oc] - df_train[med_col]) / df_train[range_col]
			df_test[medrat_col] = (df_test[oc] - df_test[med_col]) / df_test[range_col]
			new_columns.append(medrat_col)
			
			minrat_col = "%s_minrat_on_%s" % (oc, gc)
			df_train[minrat_col] = (df_train[oc] - df_train[min_col]) / df_train[range_col]
			df_test[minrat_col] = (df_test[oc] - df_test[min_col]) / df_test[range_col]
			new_columns.append(minrat_col)
			
			if replace:
				drop_columns += stats_columns

		if replace:
			df_train.drop(drop_columns, axis=1, inplace=True)
			df_test.drop(drop_columns, axis=1, inplace=True)

	sys.stdout.write("\n")


	df_train, df_test, nan_columns = ImputeNanFeatures(df_train, df_test, new_columns, verbose=True)
	new_columns += nan_columns

	df_train, df_test, inf_columns = ImputeInfFeatures(df_train, df_test, new_columns, verbose=True)
	new_columns += inf_columns


	return df_train, df_test, new_columns
	# return new_columns


# def ImputeNanRatioFeatures(df_train, df_test, r_col, new_columns, verbose=False):
# 	if verbose:
# 		sys.stdout.write("Impute NaN data ... ")
# 		sys.stdout.flush()
	
# 	if df_train[r_col].isnull().sum():
# 		nan_col = r_col + '_nan'
# 		df_train[nan_col] = 0
# 		df_test[nan_col] = 0

# 		index = df_train[df_train[r_col].isnull()].index
# 		df_train.loc[index, r_col] = 0
# 		df_train.loc[index, nan_col] = 1

# 		index = df_test[df_test[r_col].isnull()].index
# 		df_test.loc[index, r_col] = 0
# 		df_test.loc[index, nan_col] = 1

# 		new_columns.append(nan_col)

# 	elif df_test[r_col].isnull().sum():
# 		new_val = df_train[r_col].mean()
# 		df_test[r_col] = df_test[r_col].fillna(new_val)

# 	if verbose:
# 		sys.stdout.write("%d done.\n" % (len(new_columns)))


# 	return df_train, df_test, new_columns


def ImputeNanFeatures(df_train, df_test, in_columns, verbose=False):
	if verbose:
		sys.stdout.write("Impute NaN data ... ")
		sys.stdout.flush()
	
	nan_columns = list()

	for c in in_columns:
		if df_train[c].isnull().sum():
			nan_col = c + '_nan'
			df_train[nan_col] = 0
			df_test[nan_col] = 0

			index = df_train[df_train[c].isnull()].index
			df_train.loc[index, c] = 0
			df_train.loc[index, nan_col] = 1

			index = df_test[df_test[c].isnull()].index
			df_test.loc[index, c] = 0
			df_test.loc[index, nan_col] = 1

			nan_columns.append(nan_col)

		elif df_test[c].isnull().sum():
			new_val = df_train[c].mean()
			df_test[c] = df_test[c].fillna(new_val)

	if verbose:
		sys.stdout.write("%d done.\n" % (len(nan_columns)))


	return df_train, df_test, nan_columns



def CreateStatsFeatures(df_train, df_test, observe_columns, group_columns):

	total_stats_mat_tr = np.array([])
	total_stats_mat_te = np.array([])
	total_stats_columns = list()

	n_grp = len(group_columns)
	n_obs = len(observe_columns)

	for i, gc in enumerate(group_columns):
		# print 'group columns =', gc
		for j, oc in enumerate(observe_columns):
			if oc == gc: continue

			msg = "\rCreate statistics columns of %s (%d/%d) on %s (%d/%d)           " % (oc, j+1, n_obs, gc, i+1, n_grp)
			sys.stdout.write(msg)
			sys.stdout.flush()

			stats_mat_tr, stats_mat_te, stats_columns = get_stats(df_train, df_test, obsv_column=oc, group_column=gc)
			
			total_stats_mat_tr = np.hstack((total_stats_mat_tr, stats_mat_tr)) if total_stats_mat_tr.size else stats_mat_tr
			total_stats_mat_te = np.hstack((total_stats_mat_te, stats_mat_te)) if total_stats_mat_te.size else stats_mat_te
			total_stats_columns += stats_columns

	sys.stdout.write("\n")

	df_stats_tr = pd.DataFrame(total_stats_mat_tr, columns=total_stats_columns, index=df_train.index)
	df_train[total_stats_columns] = df_stats_tr

	df_stats_te = pd.DataFrame(total_stats_mat_te, columns=total_stats_columns, index=df_test.index)
	df_test[total_stats_columns] = df_stats_te

	df_train[total_stats_columns] = df_train[total_stats_columns].fillna(0)
	df_test[total_stats_columns] = df_test[total_stats_columns].fillna(0)

	return df_train, df_test, total_stats_columns



def get_stats(df_train, df_test, obsv_column, group_column):

	'''
	obsv_column: numeric columns to group with (e.g. price, bedrooms, bathrooms)
	group_column: categorical columns to group on (e.g. manager_id, building_id)
	'''

	assert(obsv_column != group_column)

	df_train['row_id'] = range(df_train.shape[0])
	df_test['row_id'] = range(df_test.shape[0])
	df_train['train'] = 1
	df_test['train'] = 0
	all_df = df_train[['row_id', 'train', obsv_column, group_column]].append(df_test[['row_id','train', obsv_column, group_column]])

	all_df[obsv_column] = all_df[obsv_column].astype(float)
	all_df[group_column] = all_df[group_column].astype(str)

	grouped = all_df[[obsv_column, group_column]].groupby(group_column)

	the_size = pd.DataFrame(grouped.size()).reset_index()
	the_size.columns = [group_column, '%s_count_on_%s' % (obsv_column, group_column)]

	the_mean = pd.DataFrame(grouped.mean()).reset_index()
	the_mean.columns = [group_column, '%s_mean_on_%s' % (obsv_column, group_column)]

	the_std = pd.DataFrame(grouped.std()).reset_index().fillna(0)
	the_std.columns = [group_column, '%s_std_on_%s' % (obsv_column, group_column)]

	the_median = pd.DataFrame(grouped.median()).reset_index()
	the_median.columns = [group_column, '%s_median_on_%s' % (obsv_column, group_column)]

	the_stats = pd.merge(the_size, the_mean)
	the_stats = pd.merge(the_stats, the_std)
	the_stats = pd.merge(the_stats, the_median)

	the_max = pd.DataFrame(grouped.max()).reset_index()
	the_max.columns = [group_column, '%s_max_on_%s' % (obsv_column, group_column)]

	the_min = pd.DataFrame(grouped.min()).reset_index()
	the_min.columns = [group_column, '%s_min_on_%s' % (obsv_column, group_column)]

	the_stats = pd.merge(the_stats, the_max)
	the_stats = pd.merge(the_stats, the_min)

	all_df = pd.merge(all_df, the_stats, how='left', on=group_column)

	selected_train = all_df[all_df['train'] == 1]
	selected_test = all_df[all_df['train'] == 0]
	selected_train.sort_values('row_id', inplace=True)
	selected_test.sort_values('row_id', inplace=True)
	selected_train.drop([obsv_column, group_column, 'row_id', 'train'], axis=1, inplace=True)
	selected_test.drop([obsv_column, group_column, 'row_id', 'train'], axis=1, inplace=True)

	stats_columns = [str(col) for col in selected_train.columns]

	return np.array(selected_train), np.array(selected_test), stats_columns










def CreateInverseColumns(df, columns):
	new_columns = list()
	for c in columns:
		dtype = df[c].dtype
		if dtype != int and dtype != float:
			continue

		inv_c = c + '_inv'
		inf_c = c + '_inv_inf'

		df[inv_c] = 1. / df[c]
		df[inf_c] = 0

		index = df[df[c] == 0].index
		df.loc[index, inv_c] = 0
		df.loc[index, inf_c] = 1

		new_columns.append(inv_c)
		new_columns.append(inf_c)

	return df, new_columns



def CreateBaBrColumns(df):
	df['babr_ratio'] = 0
	index = df[df.bedroomcnt != 0].index
	df.loc[index, 'babr_ratio'] = df.loc[index, 'bathroomcnt'] / df.loc[index, 'bedroomcnt']

	df['no_br'] = 0
	index = df[df.bedroomcnt == 0].index
	df.loc[index, 'no_br'] = 1

	df['babr_diff'] = df.bathroomcnt - df.bedroomcnt

	new_columns = ['babr_ratio', 'no_br', 'babr_diff']

	return df, new_columns














