import os, re, json 
import yaml, nltk, argparse
import sentencepiece as spm
from run import load_tokenizer
from datasets import load_dataset



def load_data(task):
    if task == 'nmt':
        data = load_dataset('wmt14', 'de-en', split='train')['translation']
        
    elif task == 'dialog':
        data = load_dataset('daily_dialog', split='train')['dialog']

    elif task == 'sum':
        data = load_dataset('cnn_dailymail', '3.0.0', split='train')

    return data


#NMT
def preprocess_nmt(orig_data, volumn=32000, min_len=10, max_len=300, max_diff=50):
    volumn_cnt = 0
    concat, processed = [], []
    
    for elem in orig_data:
        temp_dict = dict()
        src, trg = elem['en'], elem['de']
        src_len, trg_len = len(src), len(trg)

        #define filtering conditions
        min_condition = (src_len >= min_len) & (trg_len >= min_len)
        max_condition = (src_len <= max_len) & (trg_len <= max_len)
        dif_condition = abs(src_len - trg_len) < max_diff

        if max_condition & min_condition & dif_condition:
            temp_dict['src'] = src.lower()
            temp_dict['trg'] = trg.lower()
            processed.append(temp_dict)
            concat.append(src + trg)
            
            #End condition
            volumn_cnt += 1
            if volumn_cnt == volumn:
                break

    with open('data/nmt/concat.txt', 'w') as f:
        f.write('\n'.join(concat))

    return processed


#Dialog
def preprocess_dialog(orig_data, volumn=32000):
    volumn_cnt = 0
    src_list, trg_list = [], []
    concat, processed = [], []

    for dial in orig_data:
        dial_list = []
        dial_turns = len(dial)
        
        for uttr in dial:
            _uttr = re.sub(r"\s([?,.!’](?:\s|$))", r'\1', uttr)
            _uttr = re.sub(r'([’])\s+', r'\1', _uttr)
            dial_list.append(_uttr.strip().lower())
        
        if dial_turns < 2:
            continue

        elif dial_turns == 2:
            src_list.append(dial_list[0])
            trg_list.append(dial_list[1])
            continue  #To avoid duplicate on below condition

        #Incase of dial_turns is even
        elif dial_turns % 2 == 0:
            src_list.extend(dial_list[0::2])
            trg_list.extend(dial_list[1::2])

            src_list.extend(dial_list[1:-1:2])
            trg_list.extend(dial_list[2::2])
        
        #Incase of dial_turns is odds
        elif dial_turns % 2 == 1:
            src_list.extend(dial_list[0:-1:2])
            trg_list.extend(dial_list[1::2])
            
            src_list.extend(dial_list[1::2])
            trg_list.extend(dial_list[2::2])   

    assert len(src_list) == len(trg_list)
    for src, trg in zip(src_list, trg_list):
        temp_dict = dict()
        temp_dict['src'] = src
        temp_dict['trg'] = trg
        
        concat.append(src + trg)
        processed.append(temp_dict)

        #End Condition
        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
        
    with open('data/dialog/concat.txt', 'w') as f:
        f.write('\n'.join(concat))
    
    return processed


#Sum
def preprocess_sum(orig_data, volumn=32000, max_num=50, min_len=500, max_len=3000):
    volumn_cnt = 0
    concat, processed = [], []

    for elem in orig_data:
        src, trg = elem['article'], elem['highlights']

        #Filter too Short or too Long Context
        if not (min_len < len(src) < max_len):
            continue
        if len(trg) > min_len:
            continue
        
        #Filter too long Sentences 
        src_split = nltk.tokenize.sent_tokenize(src)
        if len(src_split) > max_num:
            continue
        for seq in src_split:
            if len(seq) > min_len:
                continue

        #remove unnecessary characters in trg sequence
        trg = re.sub(r'\n', ' ', trg)                 #remove \n
        trg = re.sub(r"\s([.](?:\s|$))", r'\1', trg)  #remove whitespace in front of dot

        temp_dict = dict()
        temp_dict['src'] = src_split
        temp_dict['trg'] = trg

        concat.append(src + trg)
        processed.append(temp_dict)

        volumn_cnt += 1
        if volumn_cnt == volumn:
            break
    
    with open('data/sum/concat.txt', 'w') as f:
        f.write('\n'.join(concat))
    
    return processed


def build_vocab(task):
    assert os.path.exists('config.yaml')
    with open('config.yaml', 'r') as f:
        vocab_config = yaml.load(f, Loader=yaml.FullLoader)['vocab']

    assert os.path.exists(f'data/{task}/concat.txt')
    opt = f"--input=data/{task}/concat.txt\
            --model_prefix=data/{task}/spm\
            --vocab_size={vocab_config['vocab_size']}\
            --character_coverage={vocab_config['coverage']}\
            --model_type={vocab_config['type']}\
            --pad_id={vocab_config['pad_id']} --pad_piece={vocab_config['pad_piece']}\
            --unk_id={vocab_config['unk_id']} --unk_piece={vocab_config['unk_piece']}\
            --bos_id={vocab_config['bos_id']} --bos_piece={vocab_config['bos_piece']}\
            --eos_id={vocab_config['eos_id']} --eos_piece={vocab_config['eos_piece']}"

    spm.SentencePieceTrainer.Train(opt)
    os.remove(f'data/{task}/concat.txt')



def tokenize_data(task, tokenized, tokenizer):
    tokenized_data = []
    for elem in tokenized:
        temp_dict = dict()
        
        if task == 'sum':
            temp = []
            for seq in elem['src']:
                temp.append(tokenizer.EncodeAsIds(seq))
            temp_dict['src'] = temp
        else:    
            temp_dict['src'] = tokenizer.EncodeAsIds(elem['src'])
        
        temp_dict['trg'] = tokenizer.EncodeAsIds(elem['trg'])
        tokenized_data.append(temp_dict)
    
    return tokenized_data


def save_data(task, data_obj):
    #split data into train/valid/test sets
    train, valid, test = data_obj[:-2000], data_obj[-2000:-1000], data_obj[-1000:]
    data_dict = {k:v for k, v in zip(['train', 'valid', 'test'], [train, valid, test])}

    for key, val in data_dict.items():
        with open(f'data/{task}/{key}.json', 'w') as f:
            json.dump(val, f)        
        assert os.path.exists(f'data/{task}/{key}.json')
    


def main(task):
    #Prerequisite
    os.makedirs(f'data/{task}', exist_ok=True)
    if task == 'sum':
        nltk.download('punkt')

    #Load Original Data
    orig = load_data(task)

    #PreProcess Data
    if task == 'nmt':
        processed = preprocess_nmt(orig)
    elif task == 'dialog':
        processed = preprocess_dialog(orig)
    elif task == 'sum':
        processed = preprocess_sum(orig)        

    #Build Vocab
    build_vocab(task)

    #Tokenize Datasets
    tokenizer = load_tokenizer(task)
    tokenized = tokenize_data(task, processed, tokenizer)

    #Save Data
    save_data(task, tokenized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', required=True)
    
    args = parser.parse_args()
    assert args.task in ['all', 'nmt', 'dialog', 'sum']
    
    if args.task == 'all':
        for task in ['nmt', 'dialog', 'sum']:
            main(task)
    else: 
        main(args.task)