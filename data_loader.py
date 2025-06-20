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