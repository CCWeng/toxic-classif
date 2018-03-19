import pandas as pd
import numpy as np



trn = pd.read_csv("__input/train.csv")
tst = pd.read_csv("__input/test.csv")
sub = pd.read_csv("__input/sample_submission.csv")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


vectorizer = TfidfVectorizer(min_df=200, token_pattern=r'(?u)\b\w*[a-zA-Z]\w*\b')
vectorizer.fit(trn.comment_text)

vocabulary = vectorizer.vocabulary_
len(vocabulary)
vectorizer.idf_


vec_trn = vectorizer.transform(trn.comment_text)
vec_tst = vectorizer.transform(tst.comment_text)

vocab_columns = sorted(vocabulary, key=vocabulary.get)
vocab_columns = [c+'_tfidf' for c in vocab_columns]

trn_new = pd.DataFrame(vec_trn.toarray(), columns=vocab_columns, index=trn.index)
tst_new = pd.DataFrame(vec_tst.toarray(), columns=vocab_columns, index=tst.index)


trn = pd.concat([trn, trn_new], axis=1)
# tst = pd.concat([tst, tst_new], axis=1)


del trn_new, tst_new



from string import punctuation

def count_word(in_str):
	for c in punctuation:
		if c != "'":
			in_str = in_str.replace(c, ' ')

	return len(in_str.split())


trn['word_count'] = trn.comment_text.apply(count_word)
tst['word_count'] = tst.comment_text.apply(count_word)


use_columns = list()
use_columns += vocab_columns
use_columns += ['word_count']







from sklearn.model_selection import train_test_split

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

X_trn, X_val, y_trn, y_val = train_test_split(trn[use_columns], trn['toxic'], test_size=0.3)




from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_trn, y_trn)

p_trn = model.predict(X_trn)
p_val = model.predict(X_val)
pp_trn = model.predict_proba(X_trn)[:, 1]
pp_val = model.predict_proba(X_val)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

acc_trn = accuracy_score(y_trn, p_trn)
acc_val = accuracy_score(y_val, p_val)
print acc_trn, acc_val

rep_trn = classification_report(y_trn, p_trn)
rep_val = classification_report(y_val, p_val)
print '==\nTrain:'
print rep_trn
print '==\nTest:'
print rep_val


auc_trn = roc_auc_score(y_trn, pp_trn)
auc_val = roc_auc_score(y_val, pp_val)
print auc_trn, auc_val






#== take a look at toxic data

targets = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

index_toxic = dict()
index_toxic['toxic'] = trn[trn.toxic==1].index
index_toxic['severe_toxic'] = trn[trn.severe_toxic==1].index
index_toxic['obscene'] = trn[trn.obscene==1].index
index_toxic['threat'] = trn[trn.threat==1].index
index_toxic['insult'] = trn[trn.insult==1].index
index_toxic['identity_hate'] = trn[trn.identity_hate==1].index



for name in targets:
	I = index_toxic[name]
	print '\n== %s ==' % (name)
	for i in I[:5]:
		print '-', trn.loc[i].comment_text


I1 = index_toxic['toxic'].tolist()
for toxic_type in ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
	I2 = index_toxic[toxic_type].tolist()
	contained_list = [i in I1 for i in I2]
	contained_ratio = sum(contained_list) / float(len(contained_list))
	print toxic_type, ':', contained_ratio



for n1 in targets:
	s1 = set(index_toxic[n1].tolist())
	N = float(len(s1))
	print "\n== %s ==" % n1

	s22 = set()
	for n2 in targets:
		if n2 == n1: continue
		s2 = set(index_toxic[n2].tolist())
		ratio = len(s1.intersection(s2)) / N
		print "ratio in %s : %.3f" % (n2, ratio)
		s22 = s22.union(s2)

	diff_ratio = len(s1.difference(s22)) / N
	print "ratio not in above : %.3f" % diff_ratio

		

# Non-toxic data
non_toxic_index = set(trn.index.tolist())
for toxic_index in index_toxic.values():
	non_toxic_index = non_toxic_index.difference(toxic_index)

non_toxic_index = list(non_toxic_index)


for i in non_toxic_index[:10]:
	print '-', trn.loc[i].comment_text


trn['non_toxic'] = 0
trn.loc[non_toxic_index, 'non_toxic'] = 1





#==============


hash_vectorizer = HashingVectorizer()

text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer1 = TfidfVectorizer()
vectorizer2 = TfidfVectorizer(stop_words='english')
vectorizer2 = TfidfVectorizer(analyzer='word')
# tokenize and build vocab
vectorizer1.fit(text)
vectorizer2.fit(text)
# summarize
print(vectorizer1.vocabulary_)
print(vectorizer2.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())




#== Run ROC on min_df

tpr_list, fpr_list = roc(trn, tst)

def roc(trn, tst):
	target = 'toxic'
	model = LogisticRegression()

	y_trn = trn[target]
	y_tst = tst[target]

	tpr_list = list()
	fpr_list = list()


	mdf_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001]
	df_roc = pd.DataFrame(mdf_list, columns=['mdf'])

	for mdf in mdf_list:
		print "--\nmin_df =", mdf

		trn, tst, tfidf_columns = fg2.CreateTfidfFeaturess(trn, tst, min_df=mdf)

		print 'Train data size =', trn.shape
		print 'Test data size =', tst.shape
		print '# of tfidf_columns =', len(tfidf_columns)

		use_columns = list()
		use_columns += tfidf_columns
		use_columns += wcount_columns

		X_trn = trn[use_columns]
		X_tst = tst[use_columns]

		model.fit(X_trn, y_trn)

		p_trn = model.predict(X_trn)
		p_tst = model.predict(X_tst)

		tpr = fa.true_positive_rate(y_tst, p_tst)
		fpr = fa.false_positive_rate(y_tst, p_tst)

		tpr_list.append(tpr)
		fpr_list.append(fpr)

	df_roc['tpr'] = tpr_list
	df_roc['fpr'] = fpr_list

	return tpr_list, fpr_list




pdb.run("fa.ChooseWordFlags(trn, tfidf_columns)")
df_chi2 = fa.ChooseWordFlags(trn, tfidf_columns)


for tgt in targets:
	df_chi2[tgt].sort_values(ascending=False).plot()
	plt.title(tgt)
	plt.show()



pdb.run("fa.GetBagOfWords(trn, tfidf_columns)")
bag_of_words, df_chi2 = fa.GetBagOfWords(trn, tfidf_columns)




N = len(oof_columns)
s = 50
for i in range(0, N, s):
	j = min(i+s, N-1)
	fa.PlotCorrelation(trn, oof_columns[i:j])



tmp = trn[tfidf_columns+['toxic']].corr()


tmp2 = trn[oof_columns+['toxic']].corr()
tmp2 = tmp2['toxic'].abs().sort_values(ascending=False)

tmp3 = tst[oof_columns+['toxic']].corr()
tmp3 = tmp3['toxic'].abs().sort_values(ascending=False)




# --
use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns

res1 = ml.validataion(trn, tst, use_columns, targets, model)


use_columns = list()
# use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns

res2 = ml.validataion(trn, tst, use_columns, targets, model)


use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
use_columns += word_exist_columns
# use_columns += oof_columns

res3 = ml.validataion(trn, tst, use_columns, targets, model)


use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns

res4 = ml.validataion(trn, tst, use_columns, targets, model)


use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
use_columns += word_exist_columns
use_columns += oof_columns

res5 = ml.validataion(trn, tst, use_columns, targets, model)



use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns
use_columns += smooth_columns

res6 = ml.validataion(trn, tst, use_columns, targets, model)


use_columns = list()
use_columns += tfidf_columns
# use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns
use_columns += smooth_columns

res7 = ml.validataion(trn, tst, use_columns, targets, model)



# standard scaler on word_count
use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns
use_columns += smooth_columns

res8 = ml.validataion(trn, tst, use_columns, targets, model)


# set maximum of word_count to 4
use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns
use_columns += smooth_columns

res9 = ml.validataion(trn, tst, use_columns, targets, model)


min_val = float('inf')
max_val = -float('inf')
for c in smooth_columns:
	min_val = min(min_val, trn[c].min(), tst[c].min())
	max_val = max(max_val, trn[c].max(), tst[c].max())


print min_val, max_val




# compute f-score for feature selection
f_scores = list()
for c in use_columns:
	fs = fa.Ftest(trn, 'toxic', c)
	f_scores.append(fs)


# fscore = pd.DataFrame(f_scores, index=use_columns)
fscore = pd.Series(f_scores, index=use_columns)

select_features1 = fscore[fscore > 50].index.tolist()

res10 = ml.validataion(trn, tst, select_features1, targets, model)



res11 = ml.validataion(trn, tst, ft_pr_columns, targets, model)


use_columns = list()
use_columns += tfidf_columns
use_columns += wcount_columns
# use_columns += word_exist_columns
use_columns += oof_columns
use_columns += smooth_columns
use_columns += ft_pr_columns

res12 = ml.validataion(trn, tst, use_columns, targets, model)


res13 = ml.validataion(trn, tst, new_use_columns, targets, model)





#== Try RFE 
from s


res9 = ml.validataion(trn, tst, use_columns, targets, model)

from sklearn.feature_selection import RFECV
import time

start_t = time.time()

selector = RFECV(model, step=1, cv=3)
selector = selector.fit(trn[tfidf_columns], trn['toxic'])

elapsed_t = time.time() - start_t
print "Elapsed time =", elapsed_t

keep_columns = trn[oof_columns].columns[selector.support_].tolist()

res11 = ml.validataion(trn, tst, oof_columns, targets, model)
res22 = ml.validataion(trn, tst, keep_columns, targets, model)


from sklearn.feature_selection import f_classif
F, pval = f_classif(trn[oof_columns], trn['toxic'])

fscore = pd.Series(F, index=oof_columns)
fscore_columns = fscore[fscore > 150].index.tolist()
res33 = ml.validataion(trn, tst, fscore_columns, targets, model)







pdb.run("fa.FeatureSelectGreedy(trn, model, tfidf_columns, 'toxic')")
keep_columns = fa.FeatureSelectGreedy(trn, model, tfidf_columns, 'toxic')

select_columns = fa.FeatureSelectGreedy(trn, model, use_columns, 'toxic')

select_columns = dict()
select_columns['oof'] = fa.FeatureSelectGreedy(trn, model, oof_columns, 'toxic')
select_columns['smooth'] = fa.FeatureSelectGreedy(trn, model, smooth_columns, 'toxic')
select_columns['tfidf'] = fa.FeatureSelectGreedy(trn, model, tfidf_columns, 'toxic')


use_columns2 = list()
for columns in select_columns.values():
	use_columns2 += columns


res2 = ml.validataion(trn, tst, use_columns2, targets, model)


use_columns3 = fa.FeatureSelectGreedy(trn, model, use_columns2, 'toxic')
res3 = ml.validataion(trn, tst, use_columns3, targets, model)


selector = RFECV(model, step=1, cv=3)

start_time = fa.timer(None)
selector = selector.fit(trn[oof_columns], trn['toxic'])
fa.timer(start_time)

# == Try Boruta


from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

rfc = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)

res1 = ml.validataion(trn, tst, use_columns, targets, rfc)

boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)
start_time = fa.timer(None)
boruta_selector.fit(trn[use_columns].values, trn['toxic'].values)
fa.timer(start_time)

selected_columns = trn[use_columns].columns[boruta_selector.support_]

res2 = ml.validataion(trn, tst, selected_columns, targets, rfc)








res3 = ml.validataion(trn, tst, use_columns, targets, model)
res4 = ml.validataion(trn, tst, selected_columns, targets, model)

rfc = RandomForestClassifier(n_estimators=1000, n_jobs=4, class_weight='balanced', max_depth=5)
res5 = ml.validataion(trn, tst, use_columns, targets, rfc)


rfc = RandomForestClassifier(n_estimators=1000, n_jobs=4, max_depth=5)
res6 = ml.validataion(trn, tst, use_columns, targets, rfc)


from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=200, max_depth=5)
res7 = ml.validataion(trn, tst, use_columns, targets, gbm)




## fastText

import fastText as ft

trn = trn[['comment_text'] + targets]
tst = tst[['comment_text'] + targets]

trn['comment_text'] = trn['comment_text'].apply(lambda x: unicode(x.replace('\n', ' '), 'utf-8'))
tst['comment_text'] = tst['comment_text'].apply(lambda x: unicode(x.replace('\n', ' '), 'utf-8'))

prefix = '__label__'

trn_files = dict()
tst_files = dict()
for tgt in targets:
	col = tgt + '_ft'

	trn[col] = trn[tgt].apply(lambda x: prefix + str(x))
	tst[col] = tst[tgt].apply(lambda x: prefix + str(x))

	f_trn = '__fasttext/%s_trn.txt' % tgt
	f_tst = '__fasttext/%s_tst.txt' % tgt
	trn_files[tgt] = f_trn
	tst_files[tgt] = f_tst

	trn[['comment_text', col]].to_csv(f_trn, header=False, index=False, sep=' ', encoding='utf-8')
	tst[['comment_text', col]].to_csv(f_tst, header=False, index=False, sep=' ', encoding='utf-8')


# tst_text = tst.comment_text.values.tolist()
# ft.train_supervised(tst_text)

classifiers = dict()
results = dict()
predicts = dict()

for tgt in targets:
	f_trn = trn_files[tgt]
	f_tst = tst_files[tgt]

	clf = ft.train_supervised(f_trn, label=prefix)

	classifiers[tgt] = clf
	
	results[tgt] = clf.test(f_tst)
	predicts[tgt] = clf.predict(tst.comment_text.values.tolist())


from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


for tgt in targets:
	y_tst = tst[tgt].values
	
	p_tst = predicts[tgt][0]
	p_tst = [int(p[0][-1]) for p in p_tst]

	pp_tst = predicts[tgt][1].reshape(len(y_tst), 1)

	rep_tst = classification_report(y_tst, p_tst)	
	auc_tst = roc_auc_score(y_tst, pp_tst)
		

	print "\n==\nTarget : %s" % tgt
	print '* Test:'
	print rep_tst
	print "AUC = %.5f" % auc_tst



# fasttext 2018-03-08
pdb.run("dp.CreateFasttextFiles(trn, tst, '__fasttext')")
ft_files = dp.CreateFasttextFiles(trn, tst, '__fasttext')

params = dict()
params['word_ngrams'] = [1, 2, 3]
params['epoch'] = [1, 5, 15, 25]
params['lr'] = [0.1, 0.4, 0.7, 1.0]


for tgt in targets:
	best_p_param, best_r_param = fa.GridSearchFastText(ft_files[tgt]['trn'], ft_files[tgt]['tst'], params)
	ft_params[tgt] = best_p_param


pdb.run("fg2.CreateFastTextColumns(trn, tst, ft_params)")
trn, tst, ft_lb_columns, ft_pr_columns = fg2.CreateFastTextColumns(trn, tst, ft_params)


import unicodedata
unicodedata.normalize('NFKD', title).encode('ascii','ignore')

