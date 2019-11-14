import glob
import os
import re
import spacy
import string
from collections import Counter
from datetime import datetime, timedelta

import contractions
import pandas as pd
import pronouncing
from nltk.corpus import cmudict
from nltk.corpus import sentiwordnet as swn

'''
PREVIOUS: video_to_data.py

Script to calculate a number of lexical features for each routine.

By line, the following features are calculated:

1. Guerini/Yang (2015) Persuasive Euphony Scores (Alliteration, Rhyme. From CMU dictionary)
2. Liu (2014) LIWC lexical taggers (Future, Past, etc.)
3. Ji (2016) RST Tension by sentence 
4. Bertero and Fung (2016) Structural markers (Word Length, POS proportion, Sentiment, Tempo)
5. Yang (2015) Incongruity (Word Embeddings) 
6. N-Grams 

To ensure accurate measurements, we clean the routines by 1) removing intro-outro skits and 2) cleaning non-dialogue and punctuation.

NEXT: align_subs.py
'''


# cleaning subtitles
def add_to_hash(h, other_hash):
	for k in other_hash.keys():
		h[k] = other_hash[k]
	return h

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

# for time calculations
def srt_to_seconds(time):
	t = datetime.strptime(time, '%H:%M:%S,%f')
	delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
	return delta.total_seconds()


# 1) Guerini Euphony Scores from CMU dictionary

# Alliteration - count the number of repeated prefixes appeared in a sentence and normalized by the number of total phonemes.
# Rhyme score is computed as the ratio of the number of repeated suffix to the total number of phonemes.
# Rhythmical homogeneity is measured as the ratio of the count of distinct phonemes to the total count of phonemes.

# use pronouncing package

#### Methods for Euphony Features (Guerini) - Guerini > Yang for performance acts

# Set up functions
def get_counts(l):
	h = {}
	for item in l:
		if item in h:
			h[item] += 1
		else:
			h[item] = 1
	return h


def count_matching_prefix_length(l1, l2):
	c = 0
	for i in range(min(len(l1), len(l2))):
		if l1[i] == l2[i]:
			c += 1
	else:
		return c
	return c


# For a list of lists, count the prefixes that match between any 2 pairs
def count_matching_prefixes(l):
	total_count = 0
	for i, p in enumerate(l):
		for j, p2 in enumerate(l):
			if i < j:
				total_count += count_matching_prefix_length(p, p2)
	return total_count


# For a list of lists, count the suffixes that match between any 2 pairs
def count_matching_suffixes(l):
	total_count = 0
	for i, p in enumerate(l):
		for j, p2 in enumerate(l):
			if i < j:
				total_count += count_matching_suffix_length(p, p2)
	return total_count


def count_matching_suffix_length(l1, l2):
	return count_matching_prefix_length(list(reversed(l1)), list(reversed(l2)))


# returns a list of phonemes for each word in a list
def get_phone_list(words):
	l = []
	for w in words:
		try:
			phones = pronouncing.phones_for_word(w.lower())[0]
			l.append(phones)
		except:
			l.append('')
	return l


def get_phone_count(split_phone_list):
	return sum([len(p) for p in split_phone_list])


def get_all_phones(split_phone_list):
	all_phones = []
	for p in split_phone_list:
		all_phones += p
	return all_phones


def get_distinct_phone_count(split_phone_list):
	return len(get_counts(get_all_phones(split_phone_list)).keys())


# Euphony Score Functions

def get_homogeneity_feature(split_phone_list):
	return 1 - (float(get_distinct_phone_count(split_phone_list)) / get_phone_count(split_phone_list))


def get_rhyme_feature(split_phone_list):
	repeated_suffixes = count_matching_suffixes(split_phone_list)
	return float(repeated_suffixes) / get_phone_count(split_phone_list)


def get_alliteration_feature(split_phone_list):
	repeated_prefixes = count_matching_prefixes(split_phone_list)
	return float(repeated_prefixes) / get_phone_count(split_phone_list)


def get_plosive_feature(split_phone_list):
	return float(len([p for p in get_all_phones(split_phone_list) if p in plosive_list])) / get_phone_count(split_phone_list)


# outputs euphony scores for a list of words
def get_euphony_features(words):
	phone_list = get_phone_list(words)
	split_phone_list = [p.split(' ') for p in phone_list]
	return {
		'rhyme': get_rhyme_feature(split_phone_list),
		'alliteration': get_alliteration_feature(split_phone_list),
		'homogeneity': get_homogeneity_feature(split_phone_list),
		'plosive': get_plosive_feature(split_phone_list)
	}


# 4) Structural Features, POS, Sentiment, Tempo

# Part of Speech Proportion simple calculation
def get_POS_proportions(tokens):
	POS_list = []
	for token in tokens:
		if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'PROPN']:
			POS_list.append(token.pos_)

	if not POS_list:
		return Counter([]), 1

	return Counter(POS_list), len(POS_list)


# get number of syllables
def get_syllables(tokens):
	tot_syllables = 0
	for token in tokens:
		try:
			tot_syllables += [len(list(y for y in x if y[-1].isdigit())) for x in d[str(token).lower()]][0]
		except:
			pass
	return tot_syllables


# get sentiment scores
def get_sentiment_features(tokens):
	pos_score, neg_score, obj_score, i = (0, 0, 0, 0)
	for token in tokens:
		senti = []
		if token.pos_ == 'NOUN':
			try:
				senti = swn.senti_synset(str(token.lemma_) + '.n.01')
			except:
				pass
		if token.pos_ == 'VERB':
			try:
				senti = swn.senti_synset(str(token.lemma_) + '.v.01')
			except:
				pass
		if token.pos_ == 'ADJ':
			try:
				senti = swn.senti_synset(str(token.lemma_) + '.a.01')
			except:
				pass
		if token.pos_ == 'ADV':
			try:
				senti = swn.senti_synset(str(token.lemma_) + '.r.01')
			except:
				pass

		if senti:
			pos_score += senti.pos_score()
			neg_score += senti.neg_score()
			obj_score += senti.obj_score()
			i += 1

	# check for boundary case
	if i == 0:
		return {
			'avg_pos': 0,
			'avg_neg': 0,
			'avg_obj': 0,
		}
	else:
		return {
			'avg_pos': pos_score / i,
			'avg_neg': neg_score / i,
			'avg_obj': obj_score / i
		}


# output the lexical features
def get_lexical_features(words, start_time, end_time):
	tokens = nlp(words)
	(c, POS_len) = get_POS_proportions(tokens)

	return {
		'speaking_rate': len(words.split()) / (end_time - start_time),
		'tempo': get_syllables(tokens) / (end_time - start_time), # syllables per second
		'line_len': len(words.split()),
		'avg_word_len': sum(len(word) for word in words.split()) / len(words.split()),
		'POS_prop_NOUN': c['NOUN'] / POS_len,
		'POS_prop_DET': c['DET'] / POS_len,
		'POS_prop_PROPN': c['PROPN'] / POS_len,
		'POS_prop_VERB': c['VERB'] / POS_len,
		'POS_prop_ADJ': c['ADJ'] / POS_len,
		'POS_prop_ADV': c['ADV'] / POS_len,
	}


# 5) Yang (2015) Incongruity: use SpaCy word vecs

# Incongruity Features from SpaCy
def get_incongruity_features(tokens):
	# default values
	max = 0
	min = 1

	for token1 in tokens:
		for token2 in tokens:
			if token1.similarity(token2) > max and token1.similarity(token2) != 1:
				max = token1.similarity(token2)  # furthest similarity

			if token1.similarity(token2) < min:
				min = token1.similarity(token2)  # closest similarity

	return {
		'disconnection': max,
		'repetition': min
	}


# 6) n_grams

def get_n_grams(words, n):
	text = words.lower()
	line = text.split(' ')
	n_grams = []
	for i in range(n - 1, len(line)):
		n_grams.append(" ".join(line[i - n + 1:i + 1]))
	return {
		'n_grams': n_grams
	}


def compute_sub_features(words, start_time, end_time):
	tokens = nlp(words)

	# 1) Guerini euphony features
	euphony = get_euphony_features(tokens)

	# 4) Bertero and Fung markers
	sentiment = get_sentiment_features(tokens)
	lexical = get_lexical_features(words, srt_to_seconds(start_time), srt_to_seconds(end_time))

	# 5) Yang Incongruity
	incongruity = get_incongruity_features(tokens)

	# 6) n-grams
	n_grams = get_n_grams(words, 2)

	tries = 0
	feats = None
	while tries < 10 and feats is None:
		try:
			feats = add_to_hash(euphony, sentiment)
			feats = add_to_hash(feats, lexical)
			feats = add_to_hash(feats, incongruity)
			feats = add_to_hash(feats, n_grams)
		except:
			tries += 1
			feats = None

	return feats


def main():
	dir = '/Volumes/LaCie/temp/'

	plosive_list = ['P', 'T', 'K', 'B', 'D', 'G']
	d = cmudict.dict()
	nlp = spacy.load('en_core_web_lg')

	# iterate
	dirs = os.listdir(dir)
	for folder in dirs:
		os.chdir(dir + folder)
		print(folder)
		csvs = glob.glob('*.{}'.format('csv'))
		csvs.sort()
		subs = pd.read_csv(csvs[0])  # subtitle
		subs = remove_html_unicode_laughs(remove_intro_outro(subs))

		# filling up dictionary of lexical features
		features = []
		for ind, row in subs.iterrows():
			features.append(compute_sub_features(row.line, row.start_time, row.end_time))

		# concating features to data
		df = pd.DataFrame(features)
		df = df.set_index(subs.index)
		out = pd.concat([subs, df], axis=1)

		out.to_csv(os.path.join(dir, folder, folder) + '_lex.csv')

# if __name__ == '__main__':
# main()
