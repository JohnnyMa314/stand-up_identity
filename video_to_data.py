import csv
import os
import re
import shutil
import subprocess

import pandas as pd
from langdetect import detect

'''
This script takes a folder of stand-up comedy routine videos, extracts and separates the audio, and outputs the result of the laughter detection algorithm.

Makes good use of ffmpeg for video and audio extraction and processing.

The following are the steps in this data pipeline. Each step is run on the entire folder of routines, iteratively through each routine.

1. Make main folder and sub-folders to store data by routine. Copy over the video data file (mkv, avi, mp4) and existing subs.
2. Extract the .wav audio and .srt file from the embedded video data.
3. (EXTERNALLY RUN IN separate_audio.py, python 2.X) call DeepConSep to extract voice vs music (laughter) .wavs.
4. Run the laughter detection algorithm from Bamman + Gillick et al. (2018). Specify threshold and duration. Outputs laughter audio (.wav) and timestamps (.csv).
5. Transform .srt to .csv. 
6. Split audio by lines, as defined in subtitles, outputs (line_XXXX.wav) for each line.
7. Create a summary of data files to check for outliers or incompletes. 

NEXT: get_audio_features.py, get_text_features.py
'''

# specified video formats
file_types = ('.mkv', '.avi', '.mp4')


def make_folders(origin_path, data_path, file_types):
	# renaming files and replace
	for root, dirs, files in os.walk(origin_path):
		for file in files:
			if file.endswith(file_types):
				folder_name = (data_path + file.replace('-', '.').replace(' ', '.').replace('(', '.').replace(')', '.').replace('\'', ''))[
				              :-4]  # removing ext, -, ' ', for name
				# making folders for organization
				os.makedirs(folder_name, exist_ok=True)
				os.makedirs(folder_name + "/noVocals", exist_ok=True)
				os.makedirs(folder_name + "/laughs", exist_ok=True)
				os.makedirs(folder_name + "/audioLines", exist_ok=True)

				# moving video to sub folders
				shutil.copy(os.path.join(root, file), folder_name)

				print(folder_name)

				# renaming all video files
				os.rename(os.path.join(folder_name, file),
				          os.path.join(folder_name, file.replace('-', '.').replace(' ', '.').replace('(', '.').replace(')', '.').replace('\'', '')))

				# move over the subtitle file, if it exists within the video folder
				if root != origin_path:
					files = os.listdir(root)
					for f in files:
						if f.endswith('.srt'):
							shutil.copy(os.path.join(root, f), folder_name)
							os.rename(os.path.join(folder_name, f), os.path.join(folder_name, folder_name.replace(data_path, '') + '.srt'))


def extract_wav_srt(data_path, file_types):
	# filling in folders using video data
	for folder in os.listdir(data_path):
		root = os.path.join(data_path, folder)
		# checking for custom subtitle
		has_sub = False
		for file in os.listdir(root):
			if file.endswith('.srt'):
				has_sub = True
		print(root)
		for file in os.listdir(root):
			for type in file_types:
				if file.endswith(type):
					video = os.path.join(root, file)
					print(root)
					# running ffmpeg to extract .wav from video
					command = "ffmpeg -i " + video + " -y -vn -acodec pcm_s16le -ar 44100 -ac 2 " + root + '/' + file.replace(file[-4:], ".wav")
					subprocess.call(command, shell=True)

					# if doesn't have subtitle, extract .srt from video
					if not has_sub:
						command = "ffmpeg -i " + video + " -y -vn -acodec pcm_s16le -ar 44100 -ac 2 " + root + '/' + file.replace(file[-4:], ".srt")
						subprocess.call(command, shell=True)


def laughter_detector(data_path, threshold, duration):
	# run laughter detection algorithm
	for root, dirs, files in os.walk(data_path):
		for file in files:
			if file.endswith('.wav') & ("-music" in file):
				print(os.path.join(root, file))
				out = root.replace('/noVocals', '')
				print(out)
				os.system(
					'python segment_laughter.py ' + os.path.join(root, file) + ' /Volumes/LaCie/models/model.h5 ' + out + ' ' + str(threshold) + ' ' + str(
						duration))
	import datetime
	now = datetime.datetime.now()
	print("Current date and time : ")
	print(now.strftime("%Y-%m-%d %H:%M:%S"))


def srt_to_csv(data_path):
	# get csv from srt
	for root, dirs, files in os.walk(data_path):
		for file in files:
			if file.endswith('.srt'):
				print(file)
				with open(os.path.join(root, file), 'r', errors='ignore') as h:
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

				with open(root + "/" + file.replace('.srt', '.csv'), "w") as f:
					writer = csv.writer(f)
					writer.writerow(["start_time", "end_time", "line"])
					for row in rows:
						writer.writerow(row)


def split_audio_by_lines(data_path):
	for root, dirs, files in os.walk(data_path):
		for file in files:
			if file.endswith('.csv') and not file.endswith('_laughs.csv') and not file.endswith('_audio.csv') and not file.endswith('_lex.csv'):
				# read subtitles for timestamps, read audio
				subs = pd.read_csv(os.path.join(root, file))
				audio_file = os.path.join(root, file.replace('.csv', '') + '.wav')

				# list of start and end timestamps
				start_times = subs.start_time.str.replace(',', '.')
				end_times = subs.end_time.str.replace(',', '.')

				# get cut of audio by line
				for i in range(0, len(subs)):
					# ffmpeg copy audio from start_time to end_time, output into folder
					command = 'ffmpeg -y -i ' + audio_file + ' -ss ' + start_times[i] + ' -to ' + end_times[i] + ' -c copy ' + root + '/audioLines/line_' + str(
						i).zfill(4) + '.wav'
					subprocess.call(command, shell=True)


def make_checklist(data_path, out_path, video_types, audio_types):
	with open(out_path + "checklist.csv", 'w') as csvFile:
		headlist = ['filename', 'video_type', 'audio_type', 'has_subs', 'eng?', 'size', 'num_laughter']
		writer = csv.writer(csvFile)
		writer.writerow(headlist)
		for folders in os.listdir(data_path):
			print(folders)
			# check for video, audio, subtitle, laughter
			checklist = [folders, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']

			for files in os.listdir(os.path.join(data_path, folders)):
				# video check
				if files.endswith(video_types):
					checklist[1] = files[-3:]
				# audio check
				if files.endswith(audio_types):
					checklist[2] = files[-3:]
				# sub check
				if files.endswith('.srt'):
					checklist[3] = 'yes'
					# language check
					with open(os.path.join(data_path, folders, folders + '.csv'), 'r', errors='ignore') as h:
						sub = ''.join(h.readlines())
					checklist[4] = detect(sub) == 'en'
					checklist[5] = os.path.getsize(os.path.join(data_path, folders, folders + '.csv'))
				# num_laughter
				if os.path.join(data_path, folders + '/laughs'):
					checklist[6] = len(os.listdir(os.path.join(data_path, folders + '/laughs')))

			writer.writerow(checklist)


def parse_inputs():
	process = True

	try:
		in_path = sys.argv[1]
	except:
		print("Enter the audio file path as the first argument")
		process = False

	try:
		out_path = sys.argv[2]
	except:
		print("Enter the stored model path as the second argument")
		process = False

	try:
		types = sys.argv[3]
	except:
		print("Enter the output audio path as the third argument")
		process = False

	if process:
		return (in_path, out_path, types)
	else:
		return False


# Usage: python segment_laughter.py <input_audio_file> <stored_model_path> <output_folder>


if __name__ == '__main__':
	if parse_inputs():
		# origin_path, data_path, file_types = parse_inputs()
		origin_path = '/Volumes/LaCie/Netflix/'
		data_path = '/Volumes/LaCie/data/'
		out_path = '/Volumes/LaCie/'
		file_types = ('.mkv', '.avi', '.mp4')
		# make output directory folders to hold data
		make_folders(origin_path, data_path, file_types)
		extract_wav_srt(data_path, file_types)
		# laughter_detector(data_path, threshold=0.3, duration=0.3)
		laughter_detector(data_path, threshold=0.3, duration=0.7)
		srt_to_csv(data_path)
		split_audio_by_lines(data_path)
		make_checklist(data_path, out_path, video_types=file_types, audio_types=('.flac', '.wav'))
