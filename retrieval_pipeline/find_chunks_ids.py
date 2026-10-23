import argparse
import json
from datasets import load_dataset
from tqdm import tqdm


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


def process_file(input_file, output_file, text_to_id, qid_to_rels):
    """Processes input .jsonl file and finds matching chunk IDs using provided mappings."""
    matched_count = 0
    mismatched_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Processing entries"):
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            
            # Get query_id from the entry
            query_id = entry.get("query_id", "")
            rels_ids = qid_to_rels.get(query_id, [])
            
            # Process each result in the "results" list
            if "results" in entry:
                for result in entry["results"]:
                    text = result["text"]
                    
                    # Get id list for this text from text_to_id
                    candidate_ids = text_to_id.get(text, [])
                    
                    # Find matching sentence id
                    found_match = False
                    for candidate_id in candidate_ids:
                        if candidate_id in rels_ids:
                            result["id"] = candidate_id
                            found_match = True
                            matched_count += 1
                            break
                    
                    # If none of the candidates match, save first value in candidates_ids as "id" for the element
                    if not found_match:
                        if candidate_ids:
                            result["id"] = candidate_ids[0]
                            matched_count += 1
                        else:
                            result["id"] = "<UNDEFINED>" # Handle case where no candidate_ids are found
                            mismatched_count += 1
            
            # Write modified entry to output file
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete! Output saved to: {output_file}")
    print(f"Total matches: {matched_count}")
    print(f"Total mismatches: {mismatched_count}")


def main():
    parser = argparse.ArgumentParser(description="Process retrieval results and find chunk IDs.")
    parser.add_argument("--input_file", required=True, help="Path to input .jsonl file.")
    parser.add_argument("--output_file", required=True, help="Path to output .jsonl file.")
    parser.add_argument("--dataset", default="sents", help="Dataset name suffix (e.g., 'sents', 'chunks'). Default is 'sents'.")
    args = parser.parse_args()

    # Step 1-3: Load HuggingFace datasets and create mappings
    dataset_name = f"kaengreg/wikifacts-{args.dataset}"
    qrels_name = f"kaengreg/wikifacts-{args.dataset}-qrels"

    print(f"Loading {dataset_name} and creating text_to_id mapping...")
    text_to_id = load_sents_text_to_id(dataset_name, split="corpus")

    print(f"Loading {qrels_name} and creating qid_to_rels mapping...")
    qid_to_rels = load_sents_qid_to_rels(qrels_name, split="qrels")
    
    # Step 4: Process input file
    print(f"Processing {args.input_file}...")
    process_file(args.input_file, args.output_file, text_to_id, qid_to_rels)


if __name__ == "__main__":
    main()
