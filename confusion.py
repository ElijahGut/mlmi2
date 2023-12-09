import Levenshtein
import numpy as np

V = 41
vocab = []

with open('vocab_39.txt') as f:
    lines = f.readlines()
    for phn in lines:
        vocab.append(phn.strip('\n'))

def compute_confusion(predicted, ground_truth):
    C = np.zeros(V*V).reshape(V,V)
    ops = Levenshtein.editops(predicted, ground_truth)
    for op, pos_gt, pos_p in ops:   
        print(op, predicted[pos_p], ground_truth[pos_gt]) 
        gt_phn_idx = vocab.index(ground_truth[pos_gt])
        p_phn_idx = vocab.index(predicted[pos_p])
        if op == 'insert':
            C[:,-1][p_phn_idx] += 1
        elif op == 'delete':
            C[-1][p_phn_idx] += 1
        elif op == 'replace':
            C[gt_phn_idx][p_phn_idx] += 1
            C[p_phn_idx][gt_phn_idx] += 1
        else:
            print(f'op {op} is not considered in our case')
    return C


p = 'sil b iy ih sil k s ay dx ih sil d ih n sil d ow n ah sil d eh n ah f ay er s eh l f sil'.split()
gt = 'sil dh iy ih sil s k oy d ih sil d ih n sil d ow n ah sil d eh n ah f ay er s eh l f sil'.split()

p2 = 'sil dh iy ih sil k s ay dx ih sil d ih n sil d ow n ah sil d eh n ah f ay er s eh l f sil'

# [('insert', 4, 4), ('replace', 4, 5), ('replace', 16, 17), ('replace', 18, 19), ('replace', 20, 21), ('insert', 23, 24), ('replace', 23, 25), ('replace', 24, 26)]

C = compute_confusion(gt,p)
print(C)
print(C[:,-1])
print(vocab)
