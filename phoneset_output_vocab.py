import matplotlib.pyplot as plt
import json


def get_vocab(fname):
	vocab = set(['_'])
	with open(fname, 'r') as f:
		while line := f.readline():
			if len(line.split(':')[1].strip('\n ')) > 0:
				vocab.add(line.split(':')[1].strip('\n '))
	return vocab



def write_vocab(fname):
	vocab = get_vocab(fname)
	with open('vocab_39.txt', 'w+') as f:
		for v in vocab:
			f.write(v+'\n')	

	f.close()

# write_vocab('phone_map')


# visualise phone freqs
def visualise_phone_frequencies():
	train_file = 'train.json'
	N = 0
	phn_count_dict = {}	
	
	with open('vocab_39.txt') as f:
		while line := f.readline():
			phn = line.strip()
			phn_count_dict[phn] = 0
        
	with open('train.json', 'r') as g:
                fbank_data = json.load(g)

	for key in fbank_data:
		fbank_obj = fbank_data[key]
		phones = fbank_obj['phn']
		for phn in phones.split():
			phn_count_dict[phn] += 1
			N += 1
		
	f.close()
	g.close()


	sorted_items = sorted(phn_count_dict.items(), key=lambda x: x[1], reverse=True)

	plt.bar(*zip(*sorted_items))
	plt.show()
	
visualise_phone_frequencies()
