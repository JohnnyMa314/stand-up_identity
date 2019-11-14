import pickle
import random
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
from scipy.special import gammaln
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split

from topic_modeling import make_bow

'''
PREVIOUS: data_prep.py

From large df of all routines, produce NLP pipeline based on comparing identity

1. Create a tfidf between all identities
2. Run a logistic regression on gender to identify top words

'''


def Sort(sub_li):
	return (sorted(sub_li, key=lambda x: x[1]))

# from routine data, get a long string of cleaned subtitle text
def collect_routine(data_df):
	routine = ''
	# line level cleaning
	for i, row in data_df.iterrows():
		line = data_df.loc[i, 'line']
		line = re.sub(r'({.*?})', '', line)  # remove unicode
		line = re.sub(r'(<.*?>)', '', line)  # remove unicode
		line = re.sub(r'\d', '', line)
		line = line.replace(r'♪', '').replace('-', ' ')  # removing others
		line = line.replace(u'\x07n', '')
		line = line.replace(u'\xa0', ' ')  # remove unicode

		routine += line + ' '  # undercase all words, separate sentences
	# routine level cleaning
	routine = nlp(routine)
	routine = ' '.join([token.lemma_ for token in routine if token.tag_ != 'NNP'])  # lemma, remove proper nouns
	routine = routine.lower()
	return routine


# get routine slices for line_type, gender, and race
def get_identity_routine(data, line_type, sex, race):
	routines = []
	identities = [[], []]
	for info, routine in data:
		if info.ethnicity.values == race or race == 'all':
			if info.gender.values == sex or sex == 'all':
				print(info)
				if line_type == 'all':
					routines.append(collect_routine(routine))
				if line_type == 'punchline':
					routines.append(collect_routine(routine[routine.duration > 0]))
				if line_type == 'setup':
					routines.append(collect_routine(routine[routine.duration == 0]))

				identities[0].append(info.gender.values[0])
				identities[1].append(info.ethnicity.values[0])

	return [routines, identities]


def log_dcm(counts, alpha):
	# Compute the log probability of counts under a Dirichlet-Compound Multinomial distribution with parameter alpha
	N = sum(counts)
	A = sum(alpha)
	return gammaln(N + 1) - sum(gammaln(counts + 1)) + gammaln(A) - gammaln(N + A) + sum(gammaln(counts + alpha) - gammaln(alpha))


def log_betabin(k, N, a, b):
	# Compute the log probability of k successes in N trials under a beta-binomial distribution with parameters a,b
	return log_dcm(np.array([k, N - k]), np.array([a, b]))


def betabin_cdf(k, N, a, b):
	# Compute the probability of X \leq k successes in N trials under a beta-binomial distribution with parameters a,b
	if k > 0.5 * N:
		p = 1. - sum([np.exp(log_betabin(x, N, a, b)) for x in range(k + 1, N)])
	else:
		p = sum([np.exp(log_betabin(x, N, a, b)) for x in range(k + 1)])
	return p


def printSF(x, n):
	# Print x up to n significant figures.
	nsf = -int(np.floor(np.log10(x))) + n - 1
	fmtstr = "%." + str(nsf) + "f"
	return fmtstr % (x)


def make_bow(corpus, min_df, max_df, max_features, ngrams):
	# stop words
	comedy_stop_words = ['uh', 'um', '', 'like', 'know', 'oh', 've', 'isn', 'ain', 'hmm', 'em', 'ah', 'wanna', 'gotta']
	stop_words = ENGLISH_STOP_WORDS.union(comedy_stop_words)

	vectorizer = CountVectorizer(input='content', min_df=min_df, max_df=max_df, stop_words=stop_words, max_features=max_features, ngram_range=ngrams)
	bow = vectorizer.fit_transform(corpus)
	vocab = vectorizer.get_feature_names()
	dict = vectorizer.vocabulary_

	return bow, vocab, dict, vectorizer


def LASSO_regression(list_of_routines):
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression(C=0.1, penalty="l1", tol=0.01, solver='saga')
	test = clf.fit(X, y)
	thetaLasso = clf.coef_


def get_identifying_markers(bow_iden, bow_all, vocab):
	iden_X = pd.DataFrame(bow_iden.toarray())
	iden_X.columns = vocab

	bow_count = bow_iden.toarray()
	bow_count[bow_count > 0] = 1
	iden_count = pd.DataFrame(bow_count)
	iden_count.columns = vocab

	all_X = pd.DataFrame(bow_all.toarray())
	all_X.columns = vocab

	bow_count_all = bow_all.toarray()
	bow_count_all[bow_count_all > 0] = 1
	all_count = pd.DataFrame(bow_count_all)
	all_count.columns = vocab

	values = []
	for word in vocab:
		N = len(all_X)  # number of authors
		tot = len(vocab)  # total "possible" words
		kj = iden_count.sum(axis=0, skipna=True)[word]  # "success" of word occurence
		ki = all_count.sum(axis=0, skipna=True)[word]  # total num of comedians using word
		values.append([word, betabin_cdf(kj, tot, ki, N - ki)])
		print([word, kj, tot, ki, N, betabin_cdf(kj, tot, ki, N - ki)])
	return values


# get balanced data for manual tagging.
# get 'cuts' lines from each routine until each identitiy category hit 'threshold' number of lines.
def get_manual_slices(data, cuts, threshold):
	count = Counter({'other': threshold - 1})
	random.shuffle(data)
	lines = pd.DataFrame()
	sets = []
	while count[min(count, key=count.get)] < threshold:
		count['other'] = threshold
		for identity, routine in data:
			sex = identity.gender.values[0]
			race = identity.ethnicity.values[0]
			print([sex, race])
			if count[sex] < threshold or count[race] < threshold:
				print(routine.iloc[0].setname)
				rand_int = random.randrange(cuts, len(routine))  # random 'cuts' line
				line_slice = pd.DataFrame(routine.iloc[range(rand_int - cuts, rand_int)].line)  # get lines
				# assign setname and identity tags
				line_slice['setname'] = [routine.iloc[0].setname] * cuts
				line_slice['sex'] = [sex] * cuts
				line_slice['race'] = [race] * cuts
				# increase counters for identity categories
				lines = pd.concat([lines, line_slice])
				count[identity.gender.values[0]] += cuts
				count[identity.ethnicity.values[0]] += cuts

	return lines


def run_logistic_regression(lines, classes, identity):
	comedy_stop_words = ['uh', 'um', '', 'like', 'know', 'oh', 've', 'isn', 'ain', 'hmm', 'em', 'ah', 'wanna', 'gotta']
	stop_words = ENGLISH_STOP_WORDS.union(comedy_stop_words)

	# tfidf performs the best for bag of words model.
	vectorizer = TfidfVectorizer(stop_words=stop_words)
	Xs = vectorizer.fit_transform(lines)
	ys = np.asarray(classes)
	feature_names = np.asarray(vectorizer.get_feature_names())

	if identity == 'gender':
		target_names = [
			'not gendered',
			'female',
			'male']

	if identity == 'race':
		target_names = [
			'not racial',
			'nonwhite',
			'white']

	# Split the dataset in train-test 80-20
	X_train, X_test, y_train, y_test = train_test_split(
		Xs, ys, test_size=0.2, random_state=0)

	# run CV with precision scorer function, focus on positive identity tagging.
	scoring = make_scorer(precision_score, labels=[1, 2], average='macro')
	clf = LogisticRegressionCV(cv=3, Cs=10, multi_class='multinomial', scoring=scoring).fit(X_train, y_train)
	C = clf.C_[0]
	pred = clf.predict(X_test)

	# get some summary statistics
	print(classification_report(y_test, pred, target_names=target_names))
	print("top 10 keywords per class:")
	for i, label in enumerate(target_names):
		top10 = np.argsort(clf.coef_[i])[-10:]
		print("%s: %s" % (label, " ".join(feature_names[top10])))

	# train on entire model
	clf_out = LogisticRegression(C=C, multi_class='multinomial', penalty='l2', solver='lbfgs').fit(Xs, ys)
	return clf_out, vectorizer


def identity_tagger(routine, classifier, vectorizer, )


def main():
	dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.7-0.3/'

	# import list of routine dataframes as pickle
	with open('/Users/johnnyma/Documents/matched_data.pickle', 'rb') as f:
		data = pickle.load(f)

	nlp = spacy.load('en_core_web_sm')

	punchlines = [[x, y[y.duration > 0]] for x, y in data]

	out = get_manual_slices(punchlines, cuts=50, threshold=1000)
	out.to_csv('/Users/johnnyma/Documents/manual_data.csv')

	# running Bamman et al Beta-Binomial relative term frequency algorithm
	(bow, vocab, dict, vectorizer) = make_bow(all, min_df=1, max_df=1.0, max_features=5000, ngrams=(1, 1))
	bow_iden = vectorizer.transform(black)

	values = get_identifying_markers(bow_iden=bow_iden, bow_all=bow, vocab=vocab)

	hists = []
	for test, test1 in values:
		hists.append(test1)

	## running logistic regression classifier on data manually tagged into discrete gender x race categories
	man_data = pd.read_csv('/Users/johnnyma/Documents/manual_data_labeled.csv')
	man_data = man_data.fillna(int(0))

	# transforming gender columns into classes. 0-neither, 1-female, 2-male
	man_data.loc[man_data['male'] == 1, 'male'] = 2
	man_data['gender_class'] = man_data['female'].astype(int) + man_data['male'].astype(int)

	gender_classifier, gender_vectorizer = run_logistic_regression(man_data['line'], man_data['gender_class'], identity='gender')

	# transforming race columns into classes. 0-neither, 1-nonwhite, 2-white
	man_data.loc[man_data['white'] == 1, 'white'] = 2
	man_data['race_class'] = man_data['nonwhite'].astype(int) + man_data['white'].astype(int)

	race_classifier, race_vectorizer = run_logistic_regression(man_data['line'], man_data['race_class'], identity='race')

	lines = []
	for label, df in punchlines:
		for line in df.line.tolist():
			lines.append(line)

	gender_lines = gender_vectorizer.transform(lines)
	gender_classifier.predict_proba(race_lines)

	test = []
	ind = 0
	for x, y, z in gender_classifier.predict_proba(gender_lines):
		test.append([lines[ind], y])
		ind += 1

	out = Sort(test)
	out

# if __name__ == '__main__':
