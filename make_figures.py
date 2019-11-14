import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from topic_modeling import remove_intro_outro

'''

PREVIOUS: align_subs.py

Make a variety of graphs, looping for each routine.

Supporting functions cut up laughter duration into quantiles and calculate simple line statistics.

1. Basic scatterplot of laughter x line index
2. Box plot of quantiles by laughter x line index
3. Same plot as above, but scatterplot 
4. Histogram of # line index between laughter instances
5. Histogram of time (s) between laughter instances, by start_time
6. Histogram of time(s) of each laughter instance, end_time - start_time

Each graph is a separate function called in the __main__ of the script.
'''


# laughter by quantile, over time
def get_laughter_timeline(data_df, num_quants):
	# slicing up by laughter duration quantiles
	quants = np.linspace(1 / num_quants, 1 - 1 / num_quants, num_quants - 1)
	slices = data_df[data_df['duration'] > 0]['duration'].quantile(quants).tolist()
	slices.append(max(data_df['duration']))  # top
	slices.append(0)  # bottom
	slices.sort()

	# non laughter lines
	df = data_df[data_df['duration'] == 0]
	bars = []
	for i in range(0, len(df)):
		bars.append((df.iloc[i]['start_sec'], df.iloc[i]['end_sec'] - df.iloc[i]['start_sec'], 0))

	# getting laughter in specific quantiles
	for i in range(0, len(slices) - 1):
		df = data_df[(data_df['duration'] > slices[i]) & (data_df['duration'] <= slices[i + 1])]
		# laughter lines by quantile category
		for j in range(0, len(df)):
			bars.append((df.iloc[j]['start_sec'], df.iloc[j]['duration'], i + 1))  # start time, duration, quantile

	return bars


# make laughter scatter
def get_laugh_scatter(data_df, num_quants):
	# slicing up by laughter duration quantiles
	quants = np.linspace(1 / num_quants, 1 - 1 / num_quants, num_quants - 1)
	slices = data_df[data_df['duration'] > 0]['duration'].quantile(quants).tolist()
	slices.append(max(data_df['duration']))  # top
	slices.append(0)  # bottom
	slices.sort()

	scatters = []
	# non laughter lines
	df = data_df[data_df['duration'] == 0]
	lines = list(zip(df.index.values, [0] * len(df)))
	scatters.append(lines)

	# getting laughter in specific quantiles, then plotting broken bar
	for i in range(0, len(slices) - 1):
		df = data_df[(data_df['duration'] > slices[i]) & (data_df['duration'] <= slices[i + 1])]
		indices = list(zip(df.index.values, [i + 1] * len(df)))  # line index, quantile
		scatters.append(indices)
	# plotting split bar timeline chart

	scatters = [item for sublist in scatters for item in sublist]  # flatten list

	return scatters


# make narrative line graph on laughter duration and seconds in
def make_laugh_linegraph(data_df):
	df = data_df

	# stitching duration and line index
	series = pd.Series(np.array(df['duration']), index=df.index.values)

	return series


def duration_dist(data_df):
	dura = data_df[data_df['duration'] > 0]['duration']
	dura = pd.DataFrame(dura)
	return dura


# lines between punchline-laughter
def punchline_setup(data_df):
	df = data_df
	laugh_idxs = df.duration[df.duration > 0].index

	space = []
	for i in range(0, len(laugh_idxs)):
		if i == 0:
			space.append(laugh_idxs[0])
			continue
		space.append(laugh_idxs[i] - laugh_idxs[i - 1] - 1)
	return [laugh_idxs, space]


# time between laughter
def punchline_setup_time(laugh_df):
	df = laugh_df

	interval = []
	for i in range(0, len(laugh_df)):
		if i == 0:
			interval.append(laugh_df.loc[i, 'start'])
			continue
		interval.append(laugh_df.loc[i, 'end'] - laugh_df.loc[i - 1, 'start'])
	return interval


# create joke clusters
def joke_clusters(data_df):
	df = data_df
	joke_bin = np.array(df.duration != 0)
	cluster = []
	count = 1

	for i in range(0, len(joke_bin)):
		if i == 0:
			continue
		if joke_bin[i] == 1 and joke_bin[i - 1] == 1:
			count += 1
		else:
			if count > 1:
				cluster.append(count)
			count = 1

	return cluster


# make all the graphs from an input directory of routine data
def make_laugh_graphs(input_dir):
	for folder in os.listdir(input_dir):
		os.chdir(input_dir + folder)

		# split file name by year
		split = re.split('(\d{4})', folder)
		if len(split) > 1:
			setname = split[0] + '(' + split[1] + ')'
		setname = setname.replace('.', ' ')
		print(setname)

		# reading in data and laughs
		csvs = glob.glob('*.{}'.format('csv'))
		csvs.sort()
		data_df = pd.read_csv(csvs[0])
		laugh_df = pd.read_csv(csvs[1])
		if data_df.empty | laugh_df.empty:
			continue

		data_df = remove_intro_outro(remove_laugh_lines(data_df))

		### line-laughter scatterplot by quantile
		num_quarts = 4
		scatters = get_laugh_scatter(data_df, num_quarts)

		df = []
		# plotting for each quartile
		for i in range(0, num_quarts):
			xs = []
			for j in range(0, len(scatters)):
				if scatters[j][1] == i:
					xs.append(scatters[j][0])
					ys = [i] * len(xs)

			df.append(xs)
			plt.scatter(xs, ys)

		# labels
		plt.xlabel('Line Index')
		plt.ylabel('Quantiles')
		plt.title(setname + "- laughter bar-linegraph")
		plt.yticks(np.linspace(0, num_quarts, num_quarts + 1), ["{0:.0%}".format(i) for i in np.linspace(0, 1 - 1 / num_quarts, num_quarts)])  # labeling ticks

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_laughter bar-linegraph.png', dpi=1200)
		print("outputting bar-linegraph...")
		plt.clf()

		### box plot of the scatter data
		df = pd.DataFrame(df, index=["{0:.0%}".format(i) for i in np.linspace(0, 1 - 1 / num_quarts, num_quarts)])
		df.T.boxplot(vert=False)

		# labels
		plt.xlabel('Line Index')
		plt.ylabel('Quantiles')
		plt.title(setname + "- laughter box-linegraph")

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_laughter box-linegraph.png', dpi=1200)
		print("outputting box-linegraph...")
		plt.clf()

		### laughter timeline by quantile
		num_quarts = 4  # number of quartiles to be used
		bar_cuts = get_laughter_timeline(data_df, num_quarts)

		for i in range(0, num_quarts):
			lines = []
			for j in range(0, len(bar_cuts)):
				if bar_cuts[j][2] == i:
					lines.append((bar_cuts[j][0], bar_cuts[j][1]))
			plt.broken_barh(lines, (i, 1))

		# labels
		plt.xlabel('Time (s) since start')
		plt.ylabel('Quantiles')
		plt.title(setname + "- laughter bar-timegraph")
		plt.yticks(np.linspace(0.5, num_quarts - 0.5, num_quarts),
		           ["{0:.0%}".format(i) for i in np.linspace(0, 1 - 1 / num_quarts, num_quarts)])  # labeling ticks

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_laughter bar-timegraph.png', dpi=1200)
		print("outputting bar-timegraph...")
		plt.clf()

		### lines between laughs - histogram
		line_intervals = punchline_setup(data_df)
		plt.hist(line_intervals[1], bins=np.arange(max(line_intervals[1]) + 2) - 0.5)

		# labels
		plt.xlabel('Lines Between Laughs')
		plt.ylabel('Count')
		plt.title(setname + "- laugh intervals")
		plt.xticks(range(0, max(line_intervals[1]) + 2))

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_laugh-intervals.png', dpi=1200)
		print("outputting line interval graph...")
		plt.clf()

		### time (s) between laughs - histogram
		time_intervals = punchline_setup_time(laugh_df)
		plt.hist(time_intervals, bins=np.arange(max(time_intervals) + 2) - 0.5)

		# labels
		plt.xlabel('Time (s) Between Laughs')
		plt.ylabel('Count')
		plt.title(setname + "- time intervals")

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_time-intervals.png', dpi=1200)
		print("outputting time interval graph...")
		plt.clf()

		### clusters of laughs - hist
		cluster = joke_clusters(data_df)
		plt.hist(cluster, bins=np.arange(max(cluster) + 2) - 0.5)

		# labels
		plt.xlabel('Number of Laughs in a Row')
		plt.ylabel('Count')
		plt.title(setname + "- joke clusters")
		plt.xticks(range(0, max(cluster) + 2))

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_joke-clusters.png', dpi=1200)
		print("outputting laugh cluster graph...")
		plt.clf()

		### laugh duration frequency - hist
		dura = duration_dist(data_df)

		# interquartile rule (exaggerated) to remove outliers
		iqr = (dura.quantile(0.75) - dura.quantile(0.25)) * 1.5
		top = int((dura.quantile(0.75) + iqr) * 1.5)  # int to floor

		# plotting
		plt.hist(dura.duration, bins=np.arange(top) + 0.5, range=(0, top))

		# labels
		plt.xlabel('Laugh Duration (s)')
		plt.ylabel('Count')
		plt.title(setname + "- duration histogram")
		plt.xticks(np.arange(top))

		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_duration-hist.png', dpi=1200)
		print("outputting duration hist graph...")
		plt.clf()

		### line graph of laughter duration by line index
		durations = make_laugh_linegraph(data_df)

		# remove no laugh lines
		durations = durations[durations > 0]

		# fitting polynomial
		x = durations.index
		y = durations
		z = np.polyfit(x, y, 6)
		p = np.poly1d(z)
		p10 = np.poly1d(np.polyfit(x, y, 10))

		plt.scatter(x, y)

		# plt.plot(x, y, '.', color = 'red')
		plt.plot(x, p(x), '-', color='green')
		plt.plot(x, p10(x), '--', color='blue')
		quartile = np.percentile(y, 98)  # 90% percentile
		plt.ylim(0, quartile + 5)

		# labels
		plt.ylabel('laughter duration (sec)')
		plt.xlabel('subtitle index')
		plt.title(setname + " laugh graph")

		# outputting
		plt.savefig(input_dir + '/' + folder + '/graphs/' + setname + '_linegraph.png', dpi=1200)

		plt.clf()


def main():
	dir = '/Users/johnnyma/Documents/COMPLETE_NN_0.5-0.5/'
	# iterate
	for folder in os.listdir(dir):
		os.makedirs(dir + folder + '/graphs/', exist_ok=True)
	make_laugh_graphs(dir)


if __name__ == '__main__':
	main()
