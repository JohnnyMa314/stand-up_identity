import glob
import os

import pandas as pd

'''
PREVIOUS: align_subs.py

Generate a number of summary statistics for display and validation. Imports 'standup_NLP' cleaning functions.

For each routine, a single function produces the key and variety of summary statistics calculations.

1. Setname
2. Total Runtime
3. # line indices
4. # laughter instances
5. length (s) of laughter instances (end - start time)
6. # non-laughter
7. length (s) non-laughter instance (end - start time)
8. # lemmatized words
9. # unique lemmatized words
10. # words in punch lines (for "joke" density calculations)

outputs as a csv for each routine. 
'''


# get lemmas, remove stop words and punctuation
def process_text(routine):
	doc = nlp(routine)
	text = []
	for word in doc:
		if not word.is_stop and not word.is_punct:
			text.append(word.lemma_)
	return text


def get_sum_stat(data, setname):
	# instantiating variables of interest
	out = pd.DataFrame(
		columns=['setname', 'total_runtime', 'total_lines', 'total_laughter_inst', 'total_laughter_dur', 'total_nolaugh_inst', 'total_nolaugh_dur',
		         'total_words', 'total_unique_words', 'total_punch_words'])

	df = remove_laugh_lines(data)  # remove lines not spoken by comedian
	df_laugh = df[df['duration'] > 0]  # separate df of only laughs
	text = process_text(collect_routine(df))  # lemmas only, no stop words

	out = out.append({
		"setname": setname,
		"total_runtime": max(df['end_sec']),
		"total_lines": len(df),
		"total_laughter_inst": len(df_laugh),
		"total_laughter_dur": df_laugh['duration'].sum(),
		"total_nolaugh_inst": len(df) - len(df_laugh),
		"total_nolaugh_dur": max(df['end_sec']) - df_laugh['duration'].sum(),
		"total_words": len(text),
		"total_unique_words": len(set(text)),
		"total_punch_words": len(process_text(collect_routine(df_laugh)))
	}, ignore_index=True)

	# out.total_runtime = max(df['end_sec'])
	# out.total_lines = len(df)
	# out.total_laughter_inst = len(df_laugh)
	# out.total_laughter_dur = df_laugh['duration'].sum()
	# out.total_nolaugh_inst = out.total_lines - out.total_laughter_inst
	# out.total_nolaugh_dur = out.total_runtime - out.total_laughter_dur

	# # text totals
	# text = process_text(collect_routine(df)) # lemmas only, no stop words
	# out.total_words = len(text)
	# out.total_unique_words = len(set(text))
	# out.total_punch_words = len(process_text(collect_routine(df_laugh)))

	return out


def main():
	dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.3-0.3/'
	out_dir = '/Users/johnnyma/Documents/'
	import spacy
	nlp = spacy.load('en_core_web_sm')

	# instantiating
	sum_stats = pd.DataFrame()

	# iterating through data folders
	for folder in os.listdir(dir):
		os.chdir(os.path.join(dir, folder))
		list = glob.glob('*.{}'.format('csv'))

		for file in list:
			if file.endswith('_data.csv'):
				data = pd.read_csv(os.path.join(dir, folder, file))
				setname = file.replace('_data.csv', '')

				# get summary statistics
				stats = get_sum_stat(data, setname)
				sum_stats = sum_stats.append(stats)

				print(sum_stats)

	# associate with setname
	sum_stats = sum_stats.set_index('setname')
	sum_stats.to_csv(os.path.join(out_dir, 'sum_stats.csv'))


if __name__ == '__main__':
	main()
