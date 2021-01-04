import os
import copy
import random
import pandas as pd
from datasets import load_dataset
from datasets import DatasetDict, Dataset, ClassLabel

def split_random(indices, split=0.5):
    idxs = copy.deepcopy(indices)
    random.shuffle(idxs)
    len1 = int(split*len(idxs))
    return sorted(idxs[:len1]), sorted(idxs[len1:])

def hans_to_tsv(data_dir, out_dir):
    df_train = pd.read_csv( os.path.join(data_dir, 'heuristics_train_set.txt'), sep='\t')
    df_eval = pd.read_csv( os.path.join(data_dir, 'heuristics_evaluation_set.txt'), sep='\t')
    
    def relabel(df):
        df = df[['sentence1', 'sentence2', 'gold_label']]
        df['label'] = df['gold_label'].apply({"entailment": 0, "non-entailment": 1}.get)
        return df[['label', 'sentence1', 'sentence2']]
        
    df_train = relabel(df_train)
    df_eval = relabel(df_eval)
    
    test_idxs, valid_idxs = split_random(list(df_eval.index))
    
    df_test = df_eval.iloc[test_idxs].reset_index(drop=True)
    df_valid = df_eval.iloc[valid_idxs].reset_index(drop=True)
    
    df_train.to_csv(os.path.join(out_dir, 'train.tsv'), sep='\t', index=False)
    df_test.to_csv(os.path.join(out_dir, 'test.tsv'), sep='\t', index=False)
    df_valid.to_csv(os.path.join(out_dir, 'validation.tsv'), sep='\t', index=False)

def datasets_to_tsv(data_dir, *args):
    save_dir = os.path.join(data_dir, args[-1])
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    for k,v in load_dataset(*args).items():
        save_path = os.path.join(save_dir, k)
        dataset = pd.DataFrame(v)
        dataset.to_csv('{}.tsv'.format(save_path), sep='\t', index=False)

def standardize_tsv(data_dir, name, split):
    if split=='train':
        filename = '{}.tsv'.format(split)
    elif split=='test':
        filename = '{}{}.tsv'.format(split, '' if name!='mnli' else '_matched')
    elif split=='validation':
        filename = '{}{}.tsv'.format(split, '' if name!='mnli' else '_matched')
    else:
        raise TypeError
    
    test_file = "test_matched" if name=='mnli' else 'test'
    valid_file = "validation_matched" if name=='mnli' else 'validation'
    
    
    df = pd.read_csv(os.path.join(data_dir, name, filename), sep='\t').dropna().reset_index(drop=True)
    
    if name!='hans':
        df = df.rename(columns={'premise':'sentence1', 'hypothesis':'sentence2'})
        
    if name=='mnli':
        df = df.drop(columns=['idx'])
        
    if name=='snli':
        df = df[df['label']>0].reset_index(drop=True)
        
    
    if split=='test' or split=='validation':
        if name=='mnli' or name=='snli':
            idxs = split_random(list(df.index), split=0.3)[0]
            df = df.loc[idxs].reset_index(drop=True)
        elif name=='hans':
            idxs = split_random(list(df.index), split=0.2)[0]
            df = df.loc[idxs].reset_index(drop=True)
        
    return df[['label', 'sentence1', 'sentence2']]

def merge_nli(data_dir, *args):
    datasets = DatasetDict()
    
    for split in ['train', 'test', 'validation']:
        
        datasets[split] = Dataset.from_pandas(
            pd.concat([standardize_tsv(data_dir, name, split) for name in args]).reset_index(drop=True)
        )
        datasets[split].features['label'] = ClassLabel(num_classes=3, names=['entailment', 'neutral', 'contradiction'])        
    return datasets