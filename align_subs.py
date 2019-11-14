import glob
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

'''

PREVIOUS: get_audio_features.py

This script merges the indexed subtitles, indexed audio features, and time-stamped laughter tracks. This outputs our final dataset for each routine.

1. Merge the subtitles and audio features by simple inner join index match.

2. Align the subtitle csv and laughter csv by time. 

	Logic for merging is as follows:
	1. Match beginning of laughter time with closest subtitle start time. Merge.
	(old: if laughter starts within a threshold (1 second) of subtitle start time, move to next line).
	2. If laughter is assigned to a [laughter] line, move it back one line.

Outputs a subtitles + audio features <- laughter merged dataset. as well as original laughter data (for possible usage). 

NEXT: topic_modeling.py, gen_sum_stat.py, make_figures.py

PROBLEMS:

Occasionally subtitles are misaligned with when speaker says words. Forced-alignment possible?


'''


# transform subtitle timestamps to seconds
def srt_to_seconds(time):
	t = datetime.strptime(time, '%H:%M:%S,%f')
	delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
	return delta.total_seconds()


# custom function to align subs and laugh timestamps.
def find_nearest(array, value):
	array = np.asarray(array)
	close = array[array < value].max()
	idx1 = np.where(array == close)[0][0]

	return int(idx1)


# merge subtitles and laughter by timestamp.
def subs_laughter_merge(sub_df, laugh_df):
	sub_df['value'] = 0
	sub_df['duration'] = 0
	sub_df['tracks'] = ''
	sub_df['laugh_start'] = ''
	sub_df['laugh_end'] = ''
	# print(sub_df)
	# convert start time and end time to seconds in subtitle data
	for i, row in sub_df.iterrows():
		sub_df.loc[i, 'start_sec'] = round(srt_to_seconds((row['start_time'])), 2)
		sub_df.loc[i, 'end_sec'] = round(srt_to_seconds((row['end_time'])), 2)

	# delete laughs/music before subs
	for i, row in laugh_df.iterrows():
		if laugh_df['start'][i] < sub_df['end_sec'][0]:
			laugh_df.drop([i], inplace=True)

	# move [laugh] subtitle laughs back one line
	laugh_types = r'\[laughter\]|\[laughter\]|\[laughing\]|\[laughs\]|\[audience laughing\]|\[crowd laughing\]|' \
	              r'\[audience cheering\]|\[audience cheering and applauding\]|\[cheers and applause\]|\[applause\]|' \
	              r'\[crowd cheering\]|\[crowd cheering and applauding\]|\[crowd applauding\]|\[crowd laughs\]|\[crowd cheers\]|\[crowd_claps\]'

	laugh_subs = sub_df.line.str.contains(laugh_types, regex=True)
	indices = laugh_subs[laugh_subs].index

	# appending matching laugh data to each line
	for i, row in laugh_df.iterrows():
		index = find_nearest(sub_df['start_sec'], laugh_df['start'][i])
		# if laughter sub, attribute laugh to the previous line
		if index > 0 and index in indices:
			index = index - 1
		sub_df.loc[index, 'value'] += 1  # total number of laughs
		sub_df.loc[index, 'duration'] += laugh_df['end'][i] - laugh_df['start'][i]  # total duration of laughs
		sub_df.loc[index, 'tracks'] += laugh_df['laughs'][i].replace('/laughs/laugh_', '#').replace('.wav', '') + ', '  # associated laugh audio
		sub_df.loc[index, 'laugh_start'] += str(laugh_df['start'][i]) + ', '  # validation purposes
		sub_df.loc[index, 'laugh_end'] += str(laugh_df['end'][i]) + ', '  # validation purposes
	return sub_df


# clean up setname using torrent name
def get_setname(folder_name):
	split = re.split('(\d{4})', folder_name)
	if len(split) > 1:
		setname = split[0] + '(' + split[1] + ')'
	else:
		return (folder_name)
	setname = setname.replace('.', ' ')
	setname = setname.replace('  ', ' ')

	return setname


def main():
	directory = '/Volumes/LaCie/data/'
	out_dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.3-0.7/'
	# iterate
	list = os.listdir(directory)
	for folder in list:
		os.chdir(directory + folder)
		print(folder)
		csvs = glob.glob('*.{}'.format('csv'))
		print(csvs)
		if len(csvs) < 4:
			continue
		csvs.sort()
		subs = pd.read_csv(csvs[0])  # subtitle
		audio = pd.read_csv(csvs[1])  # audio features
		laughs = pd.read_csv(csvs[2])  # laughter times
		lex = pd.read_csv(csvs[3], index_col=0)
		if subs.empty | audio.empty | laughs.empty | lex.empty:
			continue
		sub_audio = pd.merge(subs, audio, left_index=True, right_index=True)
		sub_audio_lex = pd.concat([sub_audio, lex.reindex(sub_audio.index)], axis=1)
		sub_audio_lex = sub_audio_lex.loc[:, ~sub_audio_lex.columns.duplicated()]
		sub_audio_lex_laugh = subs_laughter_merge(sub_audio_lex, laughs)
		setname = get_setname(folder)
		os.makedirs(os.path.join(out_dir, setname), exist_ok=True)  # organize by folder
		sub_audio_lex_laugh.to_csv(os.path.join(out_dir, setname, (setname + '_data.csv')))
		laughs.to_csv(os.path.join(out_dir, setname, (setname + '_laughs.csv')))  # save laugh timestamps for further use


if __name__ == '__main__':
	main()
