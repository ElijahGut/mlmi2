import torchaudio
import json
import pathlib
import os
import torch
from torchaudio.compliance import kaldi

def extract_fbank_from_file(fname):
	fname = pathlib.Path(fname)
	json_to_write = {}	

	# open the file and load the JSON data
	with open(fname, 'r') as f:
		fbank_data = json.load(f)

	for key in fbank_data:
		key_split = key.split('_')
		
		spk_id = key_split[0]
		fbank_id = key_split[1]

		fbank_obj = fbank_data[key]
		wav_path = fbank_obj['wav']
		
		path_to_save = os.path.join(f'/rds/user/ejg84/hpc-work/MLMI2/TIMIT/data/fbanks/{fname.stem}', spk_id) 
		'''
		# load wav file 
		loaded_wav, sample_rate = torchaudio.load(wav_path)
		fbank_features = kaldi.fbank(loaded_wav)	
		
		# save fbank features to TIMIT/data/fbanks/fname/spk_id/fbank_id	
		
		if os.path.exists(path_to_save) != True:
			os.mkdir(path_to_save)			

		torch.save(fbank_features, os.path.join(path_to_save,fbank_id))
		'''

		# construct json to write
		val = {}	
		val['duration'] = fbank_obj['duration']
		val['phn'] = fbank_obj['phn'] 	
		val['spk_id'] = spk_id
		val['fbank'] = os.path.join(path_to_save, fbank_id)

		json_to_write[key] = val

	path_to_write = f'{fname.stem}_fbank.json'
	json_str = json.dumps(json_to_write)
	
	g = open(path_to_write, "w+")
	g.write(json_str)
	g.close 
			
files_to_extract = ['train.json', 'test.json', 'dev.json']

for f in files_to_extract:
	print(f'processing file {f}')
	extract_fbank_from_file(f)
	 
	
print('done')

