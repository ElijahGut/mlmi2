from collections import namedtuple
from decoder import decode
import models
import torch

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

args = {'seed': 123,
        'train_json': 'train_fbank.json',
        'val_json': 'dev_fbank.json',
        'test_json': 'test_fbank.json',
        'batch_size': 4,
        'num_layers': 2,
        'fbank_dims': 23,
        'model_dims': 128,
        'concat': 1,
        'lr': 0.5,
        'vocab': vocab,
        'report_interval': 50,
        'dropout': 0.3,
        'num_epochs': 20,
        'device': device,
        'optimiser': 'sgd',
        'grad_clip': 0.5,
        'is_bidir': True
       }

args = namedtuple('x', args)(**args)

model = models.BiLSTM(
    args.num_layers, args.fbank_dims * args.concat, args.model_dims, len(args.vocab), dropout=args.dropout, 
    is_bidir=args.is_bidir
)
num_params = sum(p.numel() for p in model.parameters())
print('Total number of model parameters is {}'.format(num_params))

## You can uncomment the following line and change model path to the model you want to decode
model_path="checkpoints/20231207_193515/model_20"

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

utt_file = 'single_utt.json'
results = decode(model, args, utt_file)
print("SUB: {:.2f}%, DEL: {:.2f}%, INS: {:.2f}%, COR: {:.2f}%, PER: {:.2f}%".format(*results))