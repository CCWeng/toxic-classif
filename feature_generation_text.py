import numpy as np
import pandas as pd
import fasttext as ft
import sys

import data_process as dp

seed = 7
num_folds = 5

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold



def read_ft_file(fpath):
	text = list()
	with open(fpath, 'r') as fh:
		for line in fh:
			lb = line[-11:]
			if lb not in ['__label__0\n', '__label__1\n']:
				print 'Error : inconsistent fasttext file format.'
			text.append(line[:-12])

	return text



def CreateFastTextColumns(trn, tst, ft_params):
	n_folds = 5

	cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=15)

	lb_columns = list()
	pr_columns = list()

	for tgt in targets:
		print "Create fasttext feature for %s" % tgt

		lb_c = tgt + '_ft'
		pr_c = tgt + '_ft_prob'

		trn[lb_c] = np.nan
		tst[lb_c] = np.nan
		lb_columns.append(lb_c)

		trn[pr_c] = np.nan
		tst[pr_c] = np.nan
		pr_columns.append(pr_c)

		epoch = ft_params[tgt]['epoch']
		n_gram = ft_params[tgt]['word_ngrams']
		lr = ft_params[tgt]['lr']

		use_columns = ['comment_text', tgt]

		sys.stdout.write(" train data ... ")
		sys.stdout.flush()
		for lrn_idx, prd_idx in cv.split(trn['comment_text'], trn[tgt]):
			lrn_I = trn.iloc[lrn_idx].index
			prd_I = trn.iloc[prd_idx].index

			ft_files = dp.CreateFasttextFiles(trn.loc[lrn_I, use_columns], trn.loc[prd_I, use_columns], '__fasttext2', [tgt], verbose=0)
			pred_text = read_ft_file(ft_files[tgt]['tst'])
			# pred_text = trn.loc[prd_I, 'comment_text'].apply(lambda x: x.decode('utf-8').encode('ascii', 'ignore').replace('\n', ' '))
			# pred_text = pred_text.values.tolist()

			clf = ft.supervised(ft_files[tgt]['trn'], 'ft_model', epoch=epoch, word_ngrams=n_gram, lr=lr, bucket=2000000)

			pred = clf.predict(pred_text)

			pred_lb = [int(p[0][-1]) for p in pred[0]]
			pred_pr = pred[1].reshape((len(pred_lb),))

			trn.loc[prd_I, lb_c] = pred_lb
			trn.loc[prd_I, pr_c] = pred_pr

			# preds = tmp[0]
			# preds = [int(p[0][-1]) for p in preds]

			# trn.loc[prd_I, lb_c] = preds

		sys.stdout.write('done.\n')
		sys.stdout.write(" test data ... ")
		sys.stdout.flush()

		ft_files = dp.CreateFasttextFiles(trn[use_columns], tst[use_columns], '__fasttext2', [tgt], verbose=0)
		pred_text = read_ft_file(ft_files[tgt]['tst'])

		clf = ft.supervised(ft_files[tgt]['trn'], 'ft_model', epoch=epoch, word_ngrams=n_gram, lr=lr, bucket=2000000)

		# preds = clf.predict(pred_text)[0]
		# preds = [int(p[0][-1]) for p in preds]
		# tst[lb_c] = preds

		pred = clf.predict(pred_text)

		pred_lb = [int(p[0][-1]) for p in pred[0]]
		pred_pr = pred[1].reshape((len(pred_lb),))

		tst[lb_c] = pred_lb
		tst[pr_c] = pred_pr

		sys.stdout.write('done.\n')

	# Normalize prob columns
	for pr_c in pr_columns:
		lb_c = pr_c[:-5]
		trn.loc[trn[lb_c] == 0, pr_c] = 1 - trn.loc[trn[lb_c] == 0, pr_c]
		tst.loc[tst[lb_c] == 0, pr_c] = 1 - tst.loc[tst[lb_c] == 0, pr_c]

	return trn, tst, lb_columns, pr_columns


def CreateWordExistFeatures(trn, tst, bag_of_words, doc_column='comment_text'):
	new_columns = list()
	for w in bag_of_words:
		new_c = 'w_' + w
		trn[new_c] = trn.comment_text.apply(lambda x: w in x).astype(int)
		tst[new_c] = tst.comment_text.apply(lambda x: w in x).astype(int)
		new_columns.append(new_c)

	return trn, tst, new_columns






def CreateTfidfFeaturess(trn, tst, vectorizer, doc_column='comment_text', min_df=0.001, postfix=''):

	# vectorizer = TfidfVectorizer(min_df=min_df, token_pattern=r'(?u)\b\w*[a-zA-Z]\w*\b')
	vectorizer.fit(trn[doc_column])

	vec_trn = vectorizer.transform(trn[doc_column])
	vec_tst = vectorizer.transform(tst[doc_column])

	new_columns = vectorizer.get_feature_names()
	new_columns = [c+'_tfidf'+postfix for c in new_columns]

	trn_new = pd.DataFrame(vec_trn.toarray(), columns=new_columns, index=trn.index)
	tst_new = pd.DataFrame(vec_tst.toarray(), columns=new_columns, index=tst.index)

	del_columns = list(set(trn.columns.tolist()).intersection(new_columns))
	if len(del_columns) > 0:
		trn.drop(del_columns, axis=1, inplace=True)

	del_columns = list(set(tst.columns.tolist()).intersection(new_columns))
	if len(del_columns) > 0:
		tst.drop(del_columns, axis=1, inplace=True)

	trn = pd.concat([trn, trn_new], axis=1)
	tst = pd.concat([tst, tst_new], axis=1)

	return trn, tst, new_columns




from string import punctuation

def count_word(in_str):
	for c in punctuation:
		if c != "'":
			in_str = in_str.replace(c, ' ')

	return len(in_str.split())




