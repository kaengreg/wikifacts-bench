from datasets import load_dataset
from typing import List, Dict, Any

def load_facts(dataset_name: str, split: str = 'queries') -> List[Dict[str, Any]]:

    ds = load_dataset(dataset_name, split)
    
    ds_dict = {}
    if 'corpus' in split:
        for record in ds['train']:
            ds_dict[record['id']] = {
                'text': record['text'],
                'linked articles': record['abstract'],
                'metadata': record['metadata']
            }
    elif 'queries' in split:
        for record in ds['train']:
            ds_dict[record['id']] = {
                'text': record['text'],
                'linked articles': record['linked articles'],
                'relevant articles': record['relevant articles'],
                'metadata': record['metadata']
            }
    return ds_dict


def load_queries(dataset_name: str, split: str):
    dataset = load_dataset(dataset_name, split)
    fact_dict = {}
    for record in dataset['train']:
        fact_dict[record['id']] = {
            'text': record['text'],
            'linked articles': record.get('linked articles', []),
            'relevant articles': record.get('relevant articles', []),
            'metadata': record.get('metadata', {})
        }
    return fact_dict


def load_corpus(dataset_name: str, split: str = 'corpus'):
    dataset = load_dataset(dataset_name, split)
    corpus = {}
    for record in dataset['train']:
        corpus[record['id']] = {
            'text': record['text'],
            'abstract': record['abstract'],
            'metadata': record['metadata']
        }
    return corpus