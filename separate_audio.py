import os

audio_dir = '~/Documents/Github/Applause/data/'
separated_audio_dir = '~/Documents/Github/Applause/data/'

# python examples/ikala/separate_ikala.py -i trump_20.wav -o ./ -m fft_1024.pkl
separate_script_path = '~/Documents/Github/DeepConvSep/examples/ikala/separate_ikala.py'
stored_model_path = '~/Documents/Github/DeepConvSep/fft_1024.pkl'


def run_separation(input_audio_file, output_dir):
	print
	"Running source separation..."

	cmd = 'python ' + separate_script_path + ' -i ' + input_audio_file + ' -o ' + output_dir + " -m " + stored_model_path
	os.system(cmd)


# print cmd

def get_files():
	dirs = os.listdir(audio_dir)
	all_files = []
	for d in dirs:
		all_files += [audio_dir + d + '/' + f for f in os.listdir(audio_dir + d)]
	return all_files


if __name__ == '__main__':
	data_path = "/Volumes/LaCie/data/"

	for root, dirs, files in os.walk(data_path):
		for file in files:
			if file.endswith('.wav') and ('NOVOCAL_' not in file) and ('laugh_' not in file) and ('-music' not in file) and ('-voice' not in file):
				print(file)
				output_dir = root + "/noVocals/"
				print(output_dir)
				run_separation(os.path.join(root, file), output_dir)
