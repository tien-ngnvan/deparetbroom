import pandas as pd
import random
import os
from tqdm import tqdm
from datasets import Dataset, load_dataset, concatenate_datasets



def load_collection(collect_en_path, collect_vi_path):
    collection_vi = {}
    with open(collect_vi_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc, _ = line.rstrip().split("\t")
            collection_vi[doc_id] = doc

    collection_en = {}
    with open(collect_en_path, encoding="utf-8") as f:
        for line in f:
            doc_id, doc, _ = line.rstrip().split("\t")
            collection_en[doc_id] = doc

    return collection_en, collection_vi


def load_queries(en_queries_path,vi_queries_path):
    queries_en = {}
    with open(en_queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries_en[query_id] = query

    queries_vi = {}
    with open(vi_queries_path, encoding="utf-8") as f:
        for line in f:
            query_id, query = line.rstrip().split("\t")
            queries_vi[query_id] = query

    return queries_en, queries_vi

def load_triplet(triplets_en_path, triplets_vi_path, remove_id = None):
    triplet_en = load_dataset('json', data_files = triplets_en_path, split='train', num_proc=4)
    triplet_en = triplet_en.rename_columns({'query' : 'query_en', 
                                            'positive':'positives_en', 
                                            'negatives':'negatives_en'})

    triplet_vi = load_dataset('json', data_files = triplets_vi_path, split='train', num_proc=4)
    triplet_vi = triplet_vi.rename_columns({'query' : 'query_vi', 
                                            'positive':'positives_vi', 
                                            'negatives':'negatives_vi'})

    triplet = concatenate_datasets([triplet_en, triplet_vi], axis=1)
    if remove_id is not None:
        for id in remove_id:
            triplet = triplet.filter(lambda example:example['query_en'] != id,
                                     num_proc=4, desc = f'Remove query id {id}')

    return triplet

def _save_en(dataset, path):
    if 'query' not in dataset.column_name:
        dataset = dataset.rename_columns({'query_en' : 'query', 
                                          'positives_en' : 'positives', 
                                          'negatives_en':'negatives'})
    dataset.to_json(path, num_proc=os.cpu_count())
    
def _save_vi(dataset, path):
    if 'query' not in dataset.column_name:
        dataset = dataset.rename_columns({'query_vi' : 'query', 
                                          'positives_vi' : 'positives', 
                                          'negatives_vi':'negatives'})
    dataset.to_json(path, num_proc=os.cpu_count(), force_ascii=False)

def save(dataset, path, key):
    for _ in range(1000):
            dataset = dataset.shuffle(random.randint(10, 10000))
            
    if key == ' test':
        en = dataset.map(remove_columns=['query_vi', 'positives_vi', 'negatives_vi'])
        _save_en(dataset=en, path=f'{path}/english/test.json')
    
        # Vietnamese
        vi = dataset.map(remove_columns=['query_en', 'positives_en', 'negatives_en'])
        _save_vi(dataset=vi, path=f'{path}/vietnamese/test.json')
        
        return
    
    # processing training and validation
    dataset = dataset.train_test_split(test_size=0.01, shuffle=True)
    
    # english
    en_train = dataset['train'].map(remove_columns=['query_vi', 'positives_vi', 'negatives_vi'], batched=True)
    _save_en(dataset=en_train, path=f'{path}/english/train.json')
    en_validation = dataset['test'].map(remove_columns=['query_vi', 'positives_vi', 'negatives_vi'], batched=True)
    _save_en(dataset=en_validation, path=f'{path}/english/validation.json')
    
    # validation
    vi_train = dataset['train'].map(remove_columns=['query_en', 'positive_en', 'negatives_en'], batched=True)
    _save_vi(dataset=vi_train, path=f'{path}/vietnamese/train.json')
    vi_validation = dataset['test'].map(remove_columns=['query_en', 'positive_en', 'negatives_en'], batched=True)
    _save_vi(dataset=vi_validation, path=f'{path}/vietnamese/validation.json')


def process_fn(example, query_en, query_vi, collection_en, collection_vi, get_k_samples=None):
    # filter samples less than k negatives
    if get_k_samples is not None and len(query_en) < get_k_samples:
        return 0
    
    # query
    try:
        qry_en = query_en[str(example['query_en'])]
        qry_vi = query_vi[str(example['query_vi'])]
    except:
        print('\n\nERROR an query: ', example['query_en'], ' not in query.csv\n\n')
        return 0

    # positive
    if isinstance(example['positives_en'], list):
        positive_en = collection_en[str(example['positives_en'][0])]
        positive_vi = collection_vi[str(example['positives_vi'][0])]
    else:
        try:
            positive_en = collection_en[str(example['positives_en'])]
            positive_vi = collection_vi[str(example['positives_vi'])]
        except:
            print("\n\nERROR an positive: ", example['positives_en'], ' not in collection\n\n')
            return 0

    # negatives
    negative_ids = random.choices(example['negatives_en'], k=get_k_samples)
    try:
        negatives_en = [collection_en[str(id)] for id in negative_ids]
        negatives_vi = [collection_vi[str(id)] for id in negative_ids]
    except:
        try:
            negative_ids = random.choices(example['negatives_en'], k=get_k_samples)

            negatives_en = [collection_en[str(id)] for id in negative_ids]
            negatives_vi = [collection_vi[str(id)] for id in negative_ids]
        except:
            print("\n\nERROR an negative: ", negatives_en, ' not in collection\n\n')
            return 0

    return {
            "query_en":qry_en, "query_vi":qry_vi,
            "positives_en":positive_en, "positives_vi":positive_vi,
            "negatives_en":negatives_en, "negatives_vi":negatives_vi,
    }


if __name__ == '__main__':
    # load collection
    collect_en_path = 'final/collections/filtered_english_collection.tsv'
    collect_vi_path = 'final/collections/filtered_vietnamese_collection.tsv'
    collection_en, collection_vi = load_collection(collect_en_path, collect_vi_path)
    
    triplet_en_path = ['mmarco/interim/english_dev.ids.json', 
                       'mmarco/interim/english_train.ids.json']
    
    triplet_vi_path = ['mmarco/interim/vietnamese_dev.ids.json', 
                       'mmarco/interim/vietnamese_train.ids.json']
    
    queries_en_path = ['mmarco/raw/queries/english_queries.dev.tsv',
                       'mmarco/raw/queries/filtered_english_queries.train.tsv']
    
    queries_vi_path = ['mmarco/raw/queries/vietnamese_queries.dev.tsv',
                       'mmarco/raw/queries/filtered_vietnamese_queries.train.tsv']

    output_path = ['test', 'train']

    
    for _, (triplet_en, triplet_vi, query_en, query_vi, key) in enumerate(
            zip(triplet_en_path, triplet_vi_path, queries_en_path, queries_vi_path, output_path)):
        triplets = load_triplet(triplet_en, triplet_vi)
        queries_en, queries_vi = load_queries(query_en, query_vi)

        print("\nDatasets info: ", triplets)
        results = []
        for example in tqdm(triplets, total=len(triplets)):
            result = process_fn(example, queries_en, queries_vi, collection_en, collection_vi)
            if result == 0:
                continue
            results.append(result)

        print("\nData len: ", len(results))
        print("\nConvert data to Huggingface datasets")
        dataset = Dataset.from_pandas(pd.DataFrame(data=results))
        
        # save sample
        print("\nSave processing . . . \n")
        save(dataset, path='data/processed', key=key)