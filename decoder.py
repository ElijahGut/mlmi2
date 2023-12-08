from jiwer import compute_measures, cer
import torch

from dataloader import get_dataloader
from utils import concat_inputs

# build global phone map
phone_map = {}

with open('phone_map') as f:
    lines = f.readlines()
    for line in lines:
        line_split = line.split(':')
        orig_phn = line_split[0]
        mapped_39_phn = line_split[1]
        phone_map[orig_phn] = mapped_39_phn.strip('\n')

def map_to_39(phns):
    mapped_phns = []
    for phn in phns:
        mapped_phn = phone_map[phn]
        mapped_phns.append(mapped_phn)
    return mapped_phns

def decode(model, args, json_file, char=False):
    idx2grapheme = {y: x for x, y in args.vocab.items()}
    test_loader = get_dataloader(json_file, 1, False)
    stats = [0., 0., 0., 0.]
    for data in test_loader:
        inputs, in_lens, trans, _ = data
        inputs = inputs.to(args.device)
        in_lens = in_lens.to(args.device)
        inputs, in_lens = concat_inputs(inputs, in_lens, factor=args.concat)
        with torch.no_grad():
            outputs = torch.nn.functional.softmax(model(inputs), dim=-1)
            outputs = torch.argmax(outputs, dim=-1).transpose(0, 1)
        outputs = [[idx2grapheme[i] for i in j] for j in outputs.tolist()]
        outputs = [[v for i, v in enumerate(j) if i == 0 or v != j[i - 1]] for j in outputs]
        outputs = [list(filter(lambda elem: elem != "_", i)) for i in outputs]
        outputs = [" ".join(i) for i in outputs]

        if '61'in args.vocab:
            outputs = [" ".join(map_to_39(i)) for i in outputs]

        if char:
            cur_stats = cer(trans, outputs, return_dict=True)
        else:
            cur_stats = compute_measures(trans, outputs)
        stats[0] += cur_stats["substitutions"]
        stats[1] += cur_stats["deletions"]
        stats[2] += cur_stats["insertions"]
        stats[3] += cur_stats["hits"]

    total_words = stats[0] + stats[1] + stats[3]
    sub = stats[0] / total_words * 100
    dele = stats[1] / total_words * 100
    ins = stats[2] / total_words * 100
    cor = stats[3] / total_words * 100
    err = (stats[0] + stats[1] + stats[2]) / total_words * 100
    return sub, dele, ins, cor, err

# test = 'sil dh iy ih sil s k oy tcl ih sil d ih n sil d ow n ah sil d eh n ah f ay er s eh el f sil'.split()
# print(phone_map)
# print(map_to_39(test))