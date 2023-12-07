import os
import re
import argparse

length_of_full_report = 406

def custom_eval(expression):
    try:
        # Attempt to evaluate the expression
        result = eval(expression)
        # Check if the result is a float with no fractional part
        if isinstance(result, float):
            if result <= 1:
                result *= 100
                return int(result) 
            return int(result)
        return result
    except Exception as e:
        print(f"Error evaluating expression: {e}")
        return None

def convert_to_dict(input_string):
    input_string = input_string.strip()
    # Remove the "Namespace(" and ")" from the string
    content = input_string[len("Namespace("):-1]

    # Split the content into key-value pairs
    pairs = [pair.strip() for pair in content.split(',')]

    # Split each pair into key and value
    key_value_pairs = [pair.split('=') for pair in pairs]

    # Create a dictionary from the key-value pairs
    result_dict = {key: custom_eval(value) for key, value in key_value_pairs}

    return result_dict

def compute_diff(dict1, dict2):
    # Find keys that are unique to each dictionary
    keys_only_in_dict1 = set(dict1.keys()) - set(dict2.keys())
    keys_only_in_dict2 = set(dict2.keys()) - set(dict1.keys())

    # Find keys that are common to both dictionaries but have different values
    different_values_keys = {key for key in set(dict1.keys()) & set(dict2.keys()) if dict1[key] != dict2[key]}

    # Create a new dictionary with differing key-value pairs
    diff_dict = {key: dict2[key] for key in different_values_keys}

    # # Add keys only present in dict1
    # diff_dict.update({key: dict1[key] for key in keys_only_in_dict1})

    # # Add keys only present in dict2
    # diff_dict.update({key: dict2[key] for key in keys_only_in_dict2})

    return diff_dict

def process_logs():
    logs = os.listdir(logs_dir)
    print(logs)
    for l in logs:
        with open(os.path.join(logs_dir,l), 'r') as f:
            try:
                lines = f.readlines()
            except:
                print(f'{l} is invalid')
                
        
        # log_args = convert_to_dict(lines[0])
    
        # see which arg differs from default i.e. what are we changing for this experiment?
        # diff = compute_diff(default_args, log_args) 
        # diff_list = list(diff.keys())

        # print(diff, diff_list)
        # diff_arg = diff_list[-1]
        # diff_arg_setting = [v for v in diff.values()][-1]
        
        # fname = f"{diff_arg}_{diff_arg_setting}"
        # # print(f'fname for experiment: {fname}')

        with open(f'outputs/{l}', 'w+') as g:
            print(f'writing outputs/{l}...')
            for line in lines:
                # Define a pattern to match the entire string
                pattern = re.compile(r'LOSS train \d+\.\d+ valid \d+\.\d+, valid PER \d+\.\d+%')

                if 'Total number of model parameters is' in line:
                    g.write(line)
                
                # Check if the string matches the pattern
                match = pattern.match(line)
                if match:
                    g.write(line)
                
            g.write(lines[-1])
        g.close()
        f.close()


default_args = convert_to_dict("Namespace(seed=123, train_json='train_fbank.json', val_json='dev_fbank.json', test_json='test_fbank.json', batch_size=4, num_layers=1, fbank_dims=23, model_dims=128, concat=1, lr=0.5, vocab='vocab_39.txt', report_interval=50, num_epochs=20, dropout=0, optimiser='sgd', clipping=None)")
logs_dir = 'logs'
process_logs()

