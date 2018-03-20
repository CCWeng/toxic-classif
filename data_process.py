import numpy as np
import pickle as pkl
import time
import os
import sys


from sklearn.model_selection import train_test_split



missing_value = -1
missing_thresh = 0.5


targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def CreateFasttextFiles(trn, tst, dir_path, targets=targets, verbose=1):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	files = dict()
	for tgt in targets:
		if verbose > 0:
			sys.stdout.write('Create fastText file for %s ... ' % tgt)
			sys.stdout.flush()

		fpath_trn = dir_path + '/' + tgt + '.trn.txt'
		fpath_tst = dir_path + '/' + tgt + '.tst.txt'

		comments_trn = trn['comment_text'].apply(lambda x: x.decode('utf-8').encode('ascii', 'ignore').replace('\n', ' '))
		comments_tst = tst['comment_text'].apply(lambda x: x.decode('utf-8').encode('ascii', 'ignore').replace('\n', ' '))

		labels_trn = trn[tgt].apply(lambda x: '__label__' + str(x))
		labels_tst = tst[tgt].apply(lambda x: '__label__' + str(x))

		write_fasttext_file(comments_trn, labels_trn, fpath_trn)
		write_fasttext_file(comments_tst, labels_tst, fpath_tst)

		fpath_trn2 = dir_path + '/' + tgt + '.trn'
		fpath_tst2 = dir_path + '/' + tgt + '.tst'

		os.system("cat %s | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > %s" % (fpath_trn, fpath_trn2))
		os.system("cat %s | sed -e \"s/\([.\!?,'/()]\)/ \1 /g\" | tr \"[:upper:]\" \"[:lower:]\" > %s" % (fpath_tst, fpath_tst2))
		os.system("rm %s" % fpath_trn)
		os.system("rm %s" % fpath_tst)

		files[tgt] = {'trn': fpath_trn2, 'tst': fpath_tst2}

		if verbose > 0:
			sys.stdout.write('done.\n')

	return files


def write_fasttext_file(comments, labels, fpath):
	with open(fpath, 'w') as fh:
		for cmt, lb in zip(comments, labels):
			line = cmt + ' ' + lb + '\n'
			fh.write(line)
	return



# def OutputFasttextFile(df, dir_path):
# 	with open(fpath, 'w') as fh:
# 		for index, row in df.iterrows():
# 			comment = row['comment_text'].replace('\n', ' ')
# 			labels = ''
# 			is_toxic = 0
# 			for tgt in targets:
# 				if row[tgt] == 1:
# 					labels += ('__label__'+tgt+' ')
# 					is_toxic = 1
# 			if is_toxic == 0:
# 				labels += '__label__non-toxic'
# 			line = comment + ' ' + labels + '\n'
# 			fh.write(line)
# 	return







def WriteDataFrameToCSV(df, path):
	prefix  = '_data/_gen/'
	postfix = '.csv.gz'

	if ~path.startswith(prefix):
		path = prefix + path

	if ~path.endswith(postfix):
		path = path + postfix

	start_time = time.time()

	df.to_csv(path
		, sep='|'
		, header=True
		, index=True
		, chunksize=100000
		, compression='gzip')

	elapsed_time = time.time() - start_time
	print "Elapse %.3f sec" % (elapsed_time)

	return


def DumpData(dat, fn):
	# fn = '_data/' + fn
	fh = open(fn, 'wb')
	pkl.dump(dat, fh)
	fh.close()
	return


def LoadData(fn):
	# fn = '_data/' + fn
	fh = open(fn, 'rb')
	dat = pkl.load(fh)
	fh.close()
	return dat



def ClassifyFeaturesByName(features):
	feature_classes = dict()
	for c in features:
		cls = c.split('_')[1]
		feature_classes[cls] = feature_classes.get(cls, list()) + [c]

	return feature_classes




def OutlierFeatureValueDetection(df, columns):
	for c in columns:
		mu = df[c].mean()
		sigma = df[c].std()

		df_mdist = (df[c].astype(float) - mu).abs() / sigma
		print c, (df_mdist > 3).sum(), df_mdist.max()

	return



def MissingValueImputation(df_train, df_test, columns):
	
	df_train['row_id'] = range(df_train.shape[0])
	df_test['row_id'] = range(df_test.shape[0])
	df_train['train'] = 1
	df_test['train'] = 0

	use_columns = ['row_id', 'train'] + columns

	df_all = df_train[use_columns].append(df_test[use_columns])
	for c in columns:
		if (df_all[c].isnull()).sum():
			impute_val = df_all[c].mean()
			df_all[c].fillna(impute_val, inplace=True)
			# df_all.loc[df_all[c] == missing_value, c] = impute_val

	df_train_new = df_all[df_all['train'] == 1]
	df_test_new = df_all[df_all['train'] == 0]

	df_train_new.sort_values(by='row_id', inplace=True)
	df_test_new.sort_values(by='row_id', inplace=True)

	df_train[columns] = df_train_new[columns]
	df_test[columns] = df_test_new[columns]

	df_train.drop(['row_id', 'train'], axis=1, inplace=True)
	df_test.drop(['row_id', 'train'], axis=1, inplace=True)

	return df_train, df_test





def TrainTestSplit(df, test_size=0.33):
	# df = df.sort_values('transactiondate')
	n_samples = df.shape[0]
	n_train = int(n_samples * (1-test_size))
	df_train = df.iloc[:n_train]
	df_test = df.iloc[n_train:]

	return df_train, df_test





def ExtractCateFeatures(in_features):
	cate_features = list()
	for f in in_features:
		# if f == 'id':
		# 	continue
		if f[-4:] == '_cat':
			cate_features.append(f)

	return cate_features



def ExtractBinFeatures(in_features):
	bin_features = list()
	for f in in_features:
		if f[-4:] == '_bin':
			bin_features.append(f)

	return bin_features 



def ExtractShortFeatures(df, in_features, missing_val=np.nan):
	num_samples = df.shape[0]

	if np.isnan(missing_val):
		df_ratio = (df[in_features].isnull()).sum().astype(float) / num_samples
	else:
		df_ratio = (df[in_features] == missing_value).sum().astype(float) / num_samples
	
	short_features = df_ratio[df_ratio > missing_thresh].index.tolist()

	return short_features



cardinality_thresh = 200

def ClassifyCateFeaturesByCardinality(df, cate_features):
	hi_card_features = list()
	lo_card_features = list()

	for f in cate_features:
		card = len(df[f].unique())
		if card > cardinality_thresh:
			hi_card_features.append(f)
		else:
			lo_card_features.append(f)

	print "The number of HIGH cardinality categorical features =", len(hi_card_features)
	print "The number of LOW cardinality categorical features =", len(lo_card_features)

	return hi_card_features, lo_card_features






















