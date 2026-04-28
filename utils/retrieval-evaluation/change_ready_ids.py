import json
from datasets import load_dataset
from tqdm import tqdm


def load_sents_id_to_text(dataset_name: str, split: str = 'corpus'):
    """Loads HF dataset and creates id -> text mapping."""
    ds = load_dataset(dataset_name, split)
    
    ds_dict = {}
    for record in ds['train']:
        _id = record['_id']
        text = record['text']
        ds_dict[_id] = text

    return ds_dict


def load_sents_text_to_id(dataset_name: str, split: str = 'corpus'):
    """Loads HF dataset and creates text -> [id0, ... idn] mapping."""
    ds = load_dataset(dataset_name, split)
    
    ds_dict = {}
    for record in ds['train']:
        text = record['text']
        _id = record['_id']
        if text not in ds_dict:
            ds_dict[text] = []
        ds_dict[text].append(_id)

    return ds_dict


def load_sents_qid_to_rels(dataset_name: str, split: str = 'qrels'):
    """Loads HF dataset and creates query_id -> [corpus_id0, ..., courpus_idn] mapping."""
    ds = load_dataset(dataset_name, split)

    qid_to_rels = {}
    for record in ds['dev']:
        query_id = record['query-id']
        corpus_id = record['corpus-id']
        if query_id not in qid_to_rels:
            qid_to_rels[query_id] = []
        qid_to_rels[query_id].append(corpus_id)

    return qid_to_rels


def main():
    # Step 1-3: Load HuggingFace datasets and create mappings
    print("Loading wikifacts-sents and creating id_to_text mapping...")
    id_to_text = load_sents_id_to_text("kaengreg/wikifacts-sents", split="corpus")

    print("Loading wikifacts-sents and creating text_to_id mapping...")
    text_to_id = load_sents_text_to_id("kaengreg/wikifacts-sents", split="corpus")

    print("Loading wikifacts-sents-qrels and creating qid_to_rels mapping...")
    qid_to_rels = load_sents_qid_to_rels("kaengreg/wikifacts-sents-qrels", split="qrels")
    
    # Step 4: Process retrieval_results_2_stage.jsonl
    print("Processing retrieval_results_2_stage.jsonl...")
    input_file = "../../heavy_artifacts/results_wikifacts-sents-v2_dev.json"
    output_file = "../../heavy_artifacts/adapted_scores_presplit.json"
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        in_data = json.load(f_in)
        out_data = {}
        
        for query_id, results in in_data.items():
            rels_ids = qid_to_rels.get(query_id, [])
            out_data[query_id] = {}

            for old_id, score in results.items():
                text = id_to_text.get(old_id, "")
                candidate_ids = text_to_id.get(text, [])

                found_match = False
                for candidate_id in candidate_ids:
                    if candidate_id in rels_ids:
                        out_data[query_id][candidate_id] = score
                        found_match = True
                        break

                if not found_match:
                    out_data[query_id][candidate_ids[0]] = score
        
        # Write modified entry to output file
        json.dump(out_data, f_out, ensure_ascii=False, indent=4)
    
    print(f"\nProcessing complete! Output saved to: {output_file}")


if __name__ == "__main__":
    main()
