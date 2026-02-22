import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from typing import List


# Configuration
MODEL_NAME = 'intfloat/multilingual-e5-large'
MAX_LEN = 256
BATCH_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_OUTPUT_DIR = 'data/vector_store/e5'
DATASET_NAME = 'kaengreg/wikifacts-sents'


def average_pool(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden_states = model_output.last_hidden_state
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(texts: List[str], prefix: str, model, tokenizer, device: str) -> np.ndarray:
    """Standalone embedding function for E5 models."""
    processed_texts = [f"{prefix}{t}" for t in texts]
    
    all_embeddings = []
    for i in tqdm(range(0, len(processed_texts), BATCH_SIZE), desc=f"Embedding {prefix.strip()}"):
        batch_texts = processed_texts[i:i + BATCH_SIZE]
        batch_dict = tokenizer(
            batch_texts, 
            max_length=MAX_LEN, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def load_hf_texts(dataset_name: str, split: str) -> List[str]:
    """Loads HF dataset and returns list of texts."""
    print(f"Loading dataset {dataset_name} (split: {split})...")
    ds = load_dataset(dataset_name, split)
    
    # Handle both DatasetDict and Dataset
    if hasattr(ds, 'keys') and 'train' in ds:
        data = ds['train']
    else:
        data = ds
    
    texts = []
    for record in tqdm(data, desc=f"Parsing {split}"):
        texts.append(record['text'])
    return texts


def process_and_save(split_name: str, prefix: str, model, tokenizer, deduplicate: bool = False):
    """Processes a split (corpus or queries) and saves to its own folder."""
    output_dir = os.path.join(BASE_OUTPUT_DIR, split_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    texts = load_hf_texts(DATASET_NAME, split_name)
    
    if deduplicate:
        print(f"Deduplicating {split_name}...")
        original_count = len(texts)
        # Use sorted list of set for deterministic order
        texts = sorted(list(set(texts)))
        print(f"Reduced {split_name} from {original_count} to {len(texts)} unique texts.")
    
    # 2. Generate embeddings
    print(f"Generating embeddings for {split_name}...")
    embeddings = get_embeddings(texts, prefix, model, tokenizer, DEVICE)
    
    # 3. Save
    emb_path = os.path.join(output_dir, 'embeddings.npy')
    meta_path = os.path.join(output_dir, 'metadata.json')
    
    print(f"Saving embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)
    
    print(f"Successfully saved {split_name} index to {output_dir}")


def main():
    # 1. Initialize Model
    print(f"Loading model {MODEL_NAME} to {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

    # 2. Process Queries
    process_and_save('queries', 'query: ', model, tokenizer, deduplicate=False)

    # 3. Process Corpus (with deduplication)
    process_and_save('corpus', 'passage: ', model, tokenizer, deduplicate=True)
    
    print("All tasks completed!")


if __name__ == "__main__":
    main()
