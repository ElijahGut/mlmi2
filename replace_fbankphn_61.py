import json
import pathlib
import os

def replace_fbank_phones():
    json_to_write = {}
    with open('train_fbank.json', 'r') as f:
        fbank_data = json.load(f)

    with open('train_p61.json', 'r') as g:
        train_61 = json.load(g)
    
    for fbank_id in fbank_data:
        fbank_obj = fbank_data[fbank_id]
        fbank_obj['phn'] = train_61[fbank_id]['phn'] 
        json_to_write[fbank_id] = fbank_obj
        
    path_to_write = 'train_fbank61.json'
    json_str = json.dumps(json_to_write)

    with open(path_to_write, 'w+') as h:
        h.write(json_str)
    
replace_fbank_phones()