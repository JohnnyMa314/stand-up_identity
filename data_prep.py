import glob
import os
import pickle
import re
import string
from collections import Counter

import contractions
import pandas as pd
from fuzzywuzzy import fuzz

'''

PREVIOUS: align_subs.py

Final cleaning steps before running regression analysis. 

1. Drop rows that are not dialogue
2. Convert bi_grams into binary variable categories, keep >threshold(25) occurences bi_grams
3. Merge comedian level statistics with routines. 
4. Create large dataset of 1) setup &2) punchline at comedian-routine-line level. 

NEXT: identity_classifier.py
'''


## routine cleaning functions

# remove non-dialogue lines, remove punctuation
def remove_html_unicode_laughs(data_df):
	cleanr = re.compile('<.*?>')
	laugh_types = re.compile(r'\[.*\]')

	for ind, row in data_df.iterrows():
		new_row = re.sub(cleanr, '', row.line)
		new_row = re.sub(laugh_types, '', new_row)
		new_row = new_row.encode('ascii', 'ignore').decode('unicode_escape')
		new_row = new_row.replace('-', ' ')  # replace hyphens with spaces, to avoid squeezing words together
		new_row = contractions.fix(new_row)  # change contractions to words
		new_row = new_row.translate(str.maketrans('', '', string.punctuation))  # removes punctuation
		new_row = new_row.replace(u'\x07n', '')
		new_row = new_row.strip()

		if not new_row or new_row.isspace():
			data_df.drop([ind], inplace=True)
		else:
			data_df.loc[ind, 'line'] = new_row

	return data_df


# remove music/skits at intro and outro
def remove_intro_outro(data_df):
	# cut off before intro end, after outro start
	intro_end = 0
	outro_start = len(data_df)

	# typical words found in intros and outros
	intro_markers = ['ladies and gentlemen', 'please welcome', 'to the stage', 'put your hands together', '♪']
	outro_markers = ['good night', 'good evening', 'have a wonderful', 'thank you so much', 'thank you', 'lovely audience', '♪']

	for i, row in data_df.iterrows():
		line = data_df.loc[i, 'line'].lower()
		for mark in intro_markers:
			if mark in line and i < 100 and (i - intro_end) < 10:  # make sure not capturing in middle
				intro_end = i

	# go backwards for outro
	for i, row in data_df[::-1].iterrows():
		line = data_df.loc[i, 'line'].lower()
		for mark in outro_markers:
			if mark in line and (i > len(data_df) - 100) and (outro_start - i) < 10:  # make sure not capturing in middle
				outro_start = i

	# drop lines
	for i, row in data_df.iterrows():
		if i < intro_end or i > outro_start:
			data_df.drop([i], inplace=True)

	return data_df


# get a Counter list of all bigrams in all routines
def get_all_bigrams(dir, threshold):
	bigram_counter = Counter()
	folder = os.listdir(dir)
	for folder in os.listdir(dir):
		os.chdir(os.path.join(dir, folder))
		files = glob.glob('*.{}'.format('csv'))

		for file in files:
			if file.endswith('_data.csv'):
				n_grams = []
				print(file)
				data = pd.read_csv(os.path.join(dir, folder, file))

				for sublist in data.n_grams:
					if str(sublist) != 'nan':
						for item in re.findall(r'\'(.*?)\'', sublist):  # transform string into list
							n_grams.append(item)

				# add to Counter
				bigram_counter += Counter(n_grams)

	# only return >threshold occurences
	bigram_counter = Counter(el for el in bigram_counter.elements() if bigram_counter[el] >= threshold)

	return bigram_counter


# generate df of bi-gram binary vars, key x routine indices.
def gen_bigram_cols(routine, bigram_list):
	# empty columns of keys in bi-gram binary vars
	df = pd.DataFrame(index=routine.index, columns=bigram_list)

	# for each row, get n_grams -> match to empty bi-gram df, set bi-gram to 1
	for ind, row in routine.iterrows():
		print(ind)
		if str(row.n_grams) != 'nan':  # check if row empty
			df.loc[ind] = 0
			n_grams = re.findall(r'\'(.*?)\'', row.n_grams)

			# match to bi-gram df by key
			for k in n_grams:
				if k in d:
					df[k].loc[ind] = 1

	#
	# vectorizer = CountVectorizer()
	# 	vectorizer.fit_transform(keys)
	# 	names = vectorizer.get_feature_names()
	# 	keys
	# 	len(list(set(names) & set(keys)))
	# 	X = vectorizer.transform(lines)
	# 	test = pd.DataFrame(X.toarray(), columns=keys)
	#

	return df


# get a list of routine dfs, add setname.
def get_routines_df(dir):
	routines = []
	for folder in os.listdir(dir):
		os.chdir(os.path.join(dir, folder))
		files = glob.glob('*.{}'.format('csv'))

		for file in files:
			if file.endswith('_data.csv'):
				print(file)
				data = pd.read_csv(os.path.join(dir, folder, file))
				data = remove_html_unicode_laughs(remove_intro_outro(data))
				data['setname'] = folder

				routines.append(data)

	return routines


# matches the setname with comedian
def fuzzy_comedian_match(routine, wiki):
	match_list = []
	comedian = ' '.join(routine.setname.iloc[0].split(' ')[0:2]).lower()
	print(comedian)
	for joker in wiki.comedian:
		match_list.append([joker, fuzz.ratio(joker, comedian)])
		match_list.sort(key=lambda x: x[1])

	# highest fuzzy string match
	comedian_match = match_list[len(match_list) - 1][0]

	routine = routine.assign(**wiki.loc[wiki['comedian'] == comedian_match].iloc[0])

	return [wiki.loc[wiki['comedian'] == comedian_match], routine]


def main():
	dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.7-0.3/'
	out_dir = '/Users/johnnyma/Documents/'

	# import list of routine dataframes as pickle
	# with open('/Users/johnnyma/Documents/data.pickle', 'rb') as f:
	#	routines = pickle.load(f)

	# get list of bi_grams across all routines, where occurences are above the threshold
	bigram_list = get_all_bigrams(dir, 25)
	keys = list(dict(bigram_list).keys())

	# get list of routine dataframes
	routines = get_routines_df(dir)

	# output to pickle
	with open('/Users/johnnyma/Documents/data.pickle', 'wb') as f:
		pickle.dump(routines, f, pickle.HIGHEST_PROTOCOL)

	# fill up bi-gram binary columns ( BETTER WAY WITH SKLEARN VECTORIZER )
	bigram_bins = []
	for routine in data:
		lines = []
		for ind, row in routine.iterrows():
			print(ind)
			if str(row.n_grams) != 'nan':  # check if row empty
				lines.append(row.line)

		print(routine.iloc[0].setname)
		out = gen_bigram_cols(routine, keys)
		out.to_csv(os.path.join(dir, routine.iloc[0].setname, (routine.iloc[0].setname + '_bigrams.csv')))

	# get key between routine and comedian data
	wiki = pd.read_csv('/Users/johnnyma/Documents/wiki_identity.csv')

	# making list of comedian info and routine dataframe
	df = []
	for routine in routines:
		df.append(fuzzy_comedian_match(routine, wiki))

	# output
	with open('/Users/johnnyma/Documents/matched_data.pickle', 'wb') as f:
		pickle.dump(df, f, pickle.HIGHEST_PROTOCOL)

# if __name__ == '__main__':
#	main()
