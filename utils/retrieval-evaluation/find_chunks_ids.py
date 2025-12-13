import json
from datasets import load_dataset
from tqdm import tqdm


def should_match_id(query_id: str, candidate_id: str) -> bool:
    """
    Determine if a candidate_id should match based on query_id.
    
    Rules:
    1. Numeric-looking query_id should only match numeric-looking candidate_id
       (not ones with prefixes like "bwc-")
    2. query_id with prefix "bwq-" should only match candidate_id with prefix "bwc-"
    
    Examples:
        query_id="2", candidate_id="1" -> True (both numeric)
        query_id="2", candidate_id="bwc-3" -> False (query numeric, candidate has prefix)
        query_id="bwq-123", candidate_id="bwc-456" -> True (both have matching prefixes)
        query_id="bwq-123", candidate_id="121" -> False (query has prefix, candidate numeric)
    """
    # Check if IDs are numeric-looking (just digits)
    query_is_numeric = query_id.isdigit()
    candidate_is_numeric = candidate_id.isdigit()
    
    # Check for prefixes
    query_has_bwq = query_id.startswith("bwq-")
    candidate_has_bwc = candidate_id.startswith("bwc-")
    
    # Rule 1: If query is numeric, candidate must also be numeric (no prefix)
    if query_is_numeric:
        return candidate_is_numeric
    
    # Rule 2: If query has bwq- prefix, candidate must have bwc- prefix
    if query_has_bwq:
        return candidate_has_bwc
    
    # For other cases, allow the match
    return True


def load_hf_data(dataset_name: str, split: str = 'corpus'):
    ds = load_dataset(dataset_name, split)
    
    ds_dict = {}
    for record in ds['train']:
        text = record['text']
        _id = record['_id']
        if text not in ds_dict:
            ds_dict[text] = []
        ds_dict[text].append(_id)

    return ds_dict


def main():
    # Step 1-3: Load HuggingFace dataset and create text_to_id mapping
    print("Loading dataset and creating text_to_id mapping...")
    text_to_id = load_hf_data("kaengreg/wikifacts-sents", split="corpus")
    
    print(f"Created mapping with {len(text_to_id)} entries")
    
    # Step 4: Process retrieval_results_2_stage.jsonl
    print("Processing retrieval_results_2_stage.jsonl...")
    input_file = "retrieval_results_2_stage_pre_split.jsonl"
    output_file = "retrieval_results_with_ids_pre_split.jsonl"
    
    matched_count = 0
    unmatched_count = 0
    total_results = 0
    multiple_ids_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc="Processing entries"):
            line = line.strip()
            if not line:
                continue
            
            entry = json.loads(line)
            
            # Get query_id from the entry
            query_id = entry.get("query_id", "")
            
            # Process each result in the "results" list
            if "results" in entry:
                new_results = []
                for result in entry["results"]:
                    total_results += 1
                    text = result["text"]
                    
                    # Find corresponding _id(s) from text_to_id
                    if text in text_to_id:
                        ids = text_to_id[text]
                        
                        # Filter IDs based on query_id matching rules
                        filtered_ids = [_id for _id in ids if should_match_id(query_id, _id)]
                        
                        if filtered_ids:
                            # Create a separate entry for each filtered ID
                            for _id in filtered_ids:
                                result_copy = result.copy()
                                result_copy["id"] = _id
                                new_results.append(result_copy)
                            matched_count += 1
                            if len(filtered_ids) > 1:
                                multiple_ids_count += 1
                        else:
                            # No matching IDs after filtering
                            unmatched_count += 1
                            result["id"] = "<FILTERED_OUT>"
                            new_results.append(result)
                            print(f"Warning: All IDs filtered out for query_id={query_id}, candidate_ids={ids}, text: {text[:100]}...")
                    else:
                        unmatched_count += 1
                        result["id"] = "<UNDEFINED>"
                        new_results.append(result)
                        print(f"Warning: Text not found in mapping: {text[:100]}...")
                
                # Replace the original results with the expanded results
                entry["results"] = new_results
            
            # Write modified entry to output file
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nProcessing complete!")
    print(f"Total results processed: {total_results}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Texts with multiple IDs: {multiple_ids_count}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

