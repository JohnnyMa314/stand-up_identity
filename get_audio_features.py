import csv
import os

import librosa
import numpy as np

# thanks jon gillick for the code
# install REAPER to extract pitch reliably
'''

PREVIOUS: video_to_data.py

Code to extract humor related audio features. Taken from Litman (2006) analysis of FRIENDS.

Uses REAPER and Librosa to extract the following:

Pitch (F0): Mean, Min, Max, Std, Range
Energy (RMS): Mean, Min, Max, Std, Range
Temporal: Duration, Internal Silence, Tempo (Tempo is computed as the total number of syllables divided by the duration of the turn)

One observation for each subtitle line, lasting the duration of the subtitle's timestamp.

Outputs the audio-features by index as a CSV within each folder.

Thanks to Jon Gillick for the code.

NEXT: align_subs.py
'''


# get pitch using REAPER
def get_pitch_tracking(audio, sr):
	r = str(np.random.randint(999999))
	librosa.output.write_wav('temp_%s.wav' % (r), audio, sr)
	reaper_cmd = "~/Documents/REAPER/build/reaper -i temp_%s.wav -f temp_%s.f0 -p temp_%s.pm -a" % (r, r, r)
	os.system(reaper_cmd)
	txt = open('temp_%s.f0' % (r)).read()
	txt = txt.split('\n')[7:][0:-1]
	split_lines = [l.split(' ') for l in txt]
	cleanup_cmd = "rm temp_%s.wav temp_%s.f0 temp_%s.pm" % (r, r, r)
	os.system(cleanup_cmd)
	return split_lines


# put into key format
def compute_audio_pitch_features(audio, sr):
	pitch_list = np.array([p[2] for p in get_pitch_tracking(audio, sr)]).astype(np.float32)
	mean_pitch = np.mean(pitch_list)
	max_pitch = np.max(pitch_list)
	min_pitch = np.min(pitch_list)
	range_pitch = max_pitch - min_pitch
	std_pitch = np.std(pitch_list)
	internal_silence = np.sum(pitch_list == -1) / float(len(pitch_list))
	return {'mean_pitch': mean_pitch, 'max_pitch': max_pitch, 'min_pitch': min_pitch, 'range_pitch': range_pitch, 'std_pitch': std_pitch,
	        'internal_silence': internal_silence}


# librosa energy functions
def compute_audio_energy_features(audio, sr):
	rmse = librosa.feature.rmse(audio, frame_length=1024)
	mean_energy = np.mean(rmse)
	min_energy = np.min(rmse)
	max_energy = np.max(rmse)
	range_energy = max_energy - min_energy
	std_energy = np.std(rmse)
	return {'mean_energy': mean_energy, 'min_energy': min_energy, 'max_energy': max_energy, 'range_energy': range_energy, 'std_energy': std_energy}


# organizing into key format
def compute_audio_features(audio, sr):
	tries = 0
	feats = None
	while tries < 10 and feats is None:
		try:
			feats = add_to_hash(compute_audio_pitch_features(audio, sr), compute_audio_energy_features(audio, sr))
		except:
			tries += 1
			feats = None
	return feats


def add_to_hash(h, d):
	for k in d.keys():
		h[k] = d[k]
	return h

def main():
	dir = '/Volumes/LaCie/temp/'

	# get column heads
	y, sr = librosa.load('/Volumes/LaCie/ex.wav')
	ex = compute_audio_features(y, sr)

	#
	list = os.listdir(dir)
	for folder in list:
		audio_path = dir + folder + '/audioLines/'
		lines = os.listdir(audio_path)
		print(dir + folder + '/' + folder + '_audio.csv')

		with open(dir + folder + '/' + folder + '_audio.csv', 'w') as csvFile:
			writer = csv.DictWriter(csvFile, ['file_index'] + [*ex])
			writer.writeheader()

			for line in lines:
				# read audio, get features
				print(audio_path + line)
				y, sr = librosa.load(audio_path + line)
				audio_feats = compute_audio_features(y, sr)
				audio_feats['file_index'] = line
				writer.writerow(audio_feats)

# if __name__ == '__main__':
#	main()
