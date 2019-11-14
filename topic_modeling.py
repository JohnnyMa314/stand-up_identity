import csv
import glob
import os
import re
import spacy

import gensim
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

'''

PREVIOUS: align_subs.py

Clean and analyze the combined sub, audio, and laughter data. 

(laugh_check) tests the laughter detection algorithm 

From raw text -> topic model with an NLP pipeline that does the following to the text:

1. Gathers a list of setnames, full routine text, and a list of lines.
	1a. Remove unicode, clean subtitle artifacts, lowercase words, separate sentences

2. Create bag of words model using sklearn CountVectorizer, convert to gensim sparse matrix (for MALLET)
	2a. Remove comedy/performance specific stop-words, 
	2b. Specify the min_df, max_df, max tokens, ngram amount taken

3. Run the MALLET model using the sparse matrix and its dictionary. Specify num_topics

4. Fit the model onto each routine, split up the text, graph the topics, and graph the PCA onto text.

'''


# back of napkin test to validate NN capturing laughter
def laugh_check(in_path, out_path):
	with open(out_path + 'laugh.check.csv', 'w') as csvFile:
		headlist = ['filename', 'laughs_captured', 'total_laughs', 'percent_captured']
		writer = csv.writer(csvFile)
		writer.writerow(headlist)

		for folder in os.listdir(in_path):
			os.chdir(in_path + folder)
			# get all complete csvs
			list = glob.glob('*.{}'.format('csv'))

			for file in list:
				if file.endswith('_data.csv'):
					print(file)
					df = pd.read_csv(file)
					# collecting all subtitle lines where audiences laugh
					laugh_types = '\\[laughter\\]|\\[laughter\\]|\\[laughing\\]|\\[laughs\\]|\\[audience laughing\\]|\\[crowd laughing\\]|' \
					              '\\[audience cheering\\]|\\[audience cheering and applauding\\]|\\[cheers and applause\\]|\\[applause\\]|' \
					              '\\[crowd cheering\\]|\\[crowd cheering and applauding\\]|\\[crowd applauding\\]|\\[crowd laughs\\]|\\[crowd_claps\\]'

					laugh_subs = df.line.str.contains(laugh_types, regex=True)
					indices = laugh_subs[laugh_subs].index

					check = []
					for number in indices:
						check.append(df.value.iloc[(number - 1)] > 0)  # want to check if previous line elicited audience laughter
					# lines that elicit audience laughter captured, success rate
					if len(check) > 0:
						top = sum(check)
						bottom = len(check)
						percent = str(top / bottom * 100) + "%"
						print(str(top / bottom * 100) + "% subtitled laughter captured")
					else:
						percent = "no laughs"
						top = 'N/A'
						bottom = 'N/A'
					writer.writerow([file, top, bottom, percent])


def line_slicer(list_of_lines, num_slices):
	list = list_of_lines
	new_list = []
	n = int(len(list) / (num_slices - 1))
	for i in range(0, len(list), n):
		# Create an index range for l of n items:
		new_list.append(''.join(list[i:i + n]))

	# if a perfect slice, then add an empty string at end
	if len(new_list) < num_slices:
		new_list.append('')

	return new_list


# makes corpus from lines, remove stop words, etc.
def collect_lines(data_df):
	line_list = []
	for i, row in data_df.iterrows():
		line = data_df.loc[i, 'line']
		line = re.sub(r'({.*?})', '', line)  # remove unicode
		line = re.sub(r'(<.*?>)', '', line)  # remove unicode
		line = line.replace(r'♪', '').replace('-', '')  # removing others
		line = line.replace(u'\x07n', '')
		line = line.replace(u'\xa0n', '')
		line_list.append(line.lower() + ' ')  # undercase all words, separate sentences
	return line_list


# makes corpus from lines, remove stop words, etc.
def collect_routine(data_df):
	routine = ''
	for i, row in data_df.iterrows():
		line = data_df.loc[i, 'line']
		line = re.sub(r'({.*?})', '', line)  # remove unicode
		line = re.sub(r'(<.*?>)', '', line)  # remove unicode
		line = re.sub(r'\d', '', line)
		line.replace(r'♪', '').replace('-', '')  # removing others
		routine += line.lower() + ' '  # undercase all words, separate sentences
	routine = routine.replace(u'\xa0n', ' ')  # remove unicode
	return routine


# remove html, unicode, and non dialogue
def remove_html_unicode_laughs(data_df):
	cleanr = re.compile('<.*?>')
	laugh_types = re.compile(r'\[.*\]')

	for ind, row in data_df.iterrows():
		new_row = re.sub(cleanr, '', row.line)
		new_row = re.sub(laugh_types, '', new_row)
		new_row = new_row.encode('ascii', 'ignore').decode('unicode_escape')

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


# makes 3 x n array of standup routine setname, corpus, and corpus by line. remove in/outro and unicode.
def make_routine_corpus(dir):
	routines = []
	for folder in os.listdir(dir):
		os.chdir(os.path.join(dir, folder))
		list = glob.glob('*.{}'.format('csv'))

		for file in list:
			if file.endswith('_data.csv'):
				setname = file.replace('_data.csv', '')
				print(setname)
				data = pd.read_csv(os.path.join(dir, folder, file))
				raw_sub = remove_html_unicode_laughs(remove_intro_outro(data))
				routine = collect_routine(raw_sub)
				line_list = collect_lines(raw_sub)

				routines.append([setname, routine, line_list])

	return routines


# make bow from sci-kit Count Vectorizer
def make_bow(corpus, min_df, max_df, max_features, ngrams):
	# stop words
	comedy_stop_words = ['uh', 'um', '', 'like', 'know', 'oh', 've', 'isn', 'ain', 'hmm', 'em', 'ah', 'wanna', 'gotta']
	stop_words = ENGLISH_STOP_WORDS.union(comedy_stop_words)

	vectorizer = CountVectorizer(input='content', min_df=min_df, max_df=max_df, stop_words=stop_words, max_features=max_features, ngram_range=(1, ngrams))
	bow = vectorizer.fit_transform(corpus)
	vocab = vectorizer.get_feature_names()
	dict = vectorizer.vocabulary_

	return bow, vocab, dict, vectorizer


# LDA
def run_LDA(sparse_bow, vocab, num_topics):
	lda = LatentDirichletAllocation(n_components=num_topics, learning_method='online')
	ldaOut = lda.fit_transform(sparse_bow)
	wordLabels = vocab

	topicLabels = []
	for i, topic in enumerate(lda.components_):
		print("Topic {}".format(i))
		topicLabel = " ".join([wordLabels[i] for i in topic.argsort()[:-10 - 1:-1]])
		print(topicLabel)
		topicLabels.append(topicLabel)

	df = pd.DataFrame(ldaOut, columns=topicLabels)

	return df


# from sci-kit Counter to gensim bow vector
def sci_to_gen(sparse_bow, dict_items):
	# transform into gensim for gensim MALLET
	corpus_vec_gen = gensim.matutils.Sparse2Corpus(sparse_bow, documents_columns=False)

	# transform scikit vocabulary into gensim dictionary
	vocab_gen = {}
	for key, val in dict_items:
		vocab_gen[val] = key

	return corpus_vec_gen, vocab_gen


# run MALLET topic modeling
def run_mallet(processed_corpus, num_topics, dictionary):
	from gensim.models.wrappers import LdaMallet
	model = LdaMallet('/Users/johnnyma/mallet-2.0.8/bin/mallet', corpus=processed_corpus, num_topics=num_topics, id2word=dictionary)

	return model


# make PCA graph of routines using spacy
def make_PCA(setnames, corpus):
	nlp = spacy.load('en_core_web_sm')

	standup_docs = [nlp(corpus) for corpus in corpus]
	setnames = setnames
	standup_vecs = [doc.vector for doc in standup_docs]

	sims = []
	for vec in standup_docs:
		thisSims = [vec.similarity(other) for other in standup_docs]
		sims.append(thisSims)

	df = pd.DataFrame(sims, columns=setnames, index=setnames)

	embedded = PCA(n_components=2).fit_transform(standup_vecs)
	embedded = embedded

	print(df[df < 1].idxmax())

	xs, ys = embedded[:, 0], embedded[:, 1]
	for i in range(len(xs)):
		plt.scatter(xs[i], ys[i])
		plt.annotate(setnames[i], (xs[i], ys[i]))

	plt.savefig('/Users/johnnyma/Documents/' + 'routine-PCA.pdf')
	print("outputting standup PCA graph...")


def graph_topics(model, vectorizer, dict, routines, num_slices, out_dir):
	# instantiating
	setnames = [set for set, text, lines in routines]
	lines = [lines for set, text, lines in routines]
	n_topics = model.num_topics

	# for every routine, create a graph
	for i in range(0, len(setnames)):
		print(setnames[i])
		groupedLines = line_slicer(lines[i], num_slices)  # concat lines into num_slices (N) groups
		routineLines = vectorizer.fit_transform(groupedLines)  # generate sparse matrix

		# change to fit MALLET format
		(line_vec, line_dict) = sci_to_gen(routineLines, dict.items())

		# get probability of each chunk
		vector = model[line_vec]

		# create figure
		plt.figure()
		fig, axs = plt.subplots(nrows=n_topics, ncols=1, sharex='all', sharey='all', constrained_layout=True)

		# graph each topic iteratively
		for j in range(0, n_topics):
			topic_labels = ', '.join([x[0] for x in model.show_topic(j, 10)])

			# making the data from applied model
			xs = range(0, num_slices)
			ys = [x[j][1] for x in vector]

			axs[j].plot(xs, ys)

			# label
			axs[j].set_title("topic " + str(j) + ": " + topic_labels, fontsize=8)
			axs[j].set_xticks(range(0, num_slices))
			axs[j].get_yaxis().set_major_locator(MultipleLocator(base=.05))

		# label axes
		plt.xlabel("chunk index")
		# fig.text(0.02, 0.5, 'topic probability', va='center', rotation='vertical')
		fig.suptitle(str(n_topics) + " topic model, " + str(setnames[i]))
		fig.set_size_inches(5, 20, forward=True)
		fig.savefig(os.path.join(out_dir, setnames[i], 'graphs') + '/' + str(n_topics) + 'topic-model.png', dpi=1200)
		plt.clf()
		fig.clf()


# run program
def main():
	dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.5-0.5/'
	out_dir = '/Users/johnnyma/Documents/'
	routines = make_routine_corpus(dir)

	# set up
	set = [set for set, text, lines in routines]
	corpus = [text for set, text, lines in routines]
	(bow, vocab, dict, vectorizer) = make_bow(corpus, min_df=2, max_df=0.90, max_features=15000, ngrams=1)

	# LDA
	lda = run_LDA(bow, vocab, num_topics=15)

	# MALLET
	(bow_vec, bow_dict) = sci_to_gen(bow, dict.items())
	model = run_mallet(processed_corpus=bow_vec, num_topics=10, dictionary=bow_dict)
	model.print_topics()

	# graph the topics, output for each comedian
	graph_topics(model, vectorizer, dict, routines, num_slices=20, out_dir=dir)

	make_PCA(set, corpus)

# if __name__ == '__main__':
#	main()
