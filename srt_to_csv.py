import csv
import os
import re

#
wav_path = "/Volumes/LaCie/raw_wav/"

for file in os.listdir(wav_path):
	if file.endswith(".srt"):
		with open(os.path.join(wav_path, file), 'r') as h:
			sub = h.readlines()

		re_pattern = r'[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}'
		regex = re.compile(re_pattern)
		# Get start times
		times = list(filter(regex.search, sub))

		start_times = list()
		end_times = list()
		for time in times:
			start_times.append(time.split(" --> ")[0])
			end_times.append(time.split(" --> ")[1][:-1])  # remove the /n

		# Get lines
		lines = [[]]
		for sentence in sub:
			if re.match(re_pattern, sentence):
				lines[-1].pop()
				lines.append([])
			else:
				lines[-1].append(sentence)
		lines = lines[1:]

		subs = list()
		for line in lines:
			subs.append(''.join(line).replace('\n', ' ').strip())  # will have some extra spaces

		# Merge results
		rows = zip(start_times, end_times, subs)

		with open(wav_path + file + ".csv", "w") as f:
			writer = csv.writer(f)
			writer.writerow(["start_time", "end_time", "line"])
			for row in rows:
				writer.writerow(row)
