import os
import sys

def friendly_print_results():
    fs = os.listdir('./outputs')
    for exp_f in fs:
        trains = []
        valids = []
        pers = []
        with open(os.path.join('outputs', exp_f), 'r') as f:
            lines = f.readlines()
        print(exp_f)
        for line in lines:
            line = line.split()
            if line[0] == 'LOSS':
                train = float(line[2].strip(',')) 
                valid = float(line[4].strip(','))
                per = float(line[7].strip('%'))
                trains.append(train)
                valids.append(valid)
                pers.append(per)
            else:
                print(' '.join(line))
        print(f'trains: {trains}')
        print(f'valids: {valids}')
        print(f'pers: {pers}')
        print()

friendly_print_results()