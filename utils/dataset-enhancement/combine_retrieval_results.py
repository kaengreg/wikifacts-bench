import json
from pathlib import Path
from collections import defaultdict


def combine_retrieval_results():
    """
    Combines retrieval results from 4 different JSONL files into a single file.
    Each entry in the combined file contains results from all 4 retrieval methods.
    """
    
    # Define the input files and their keys
    input_files = {
        'sparse_sentence': 'retrieval_results_ru_sparse_sentence.jsonl',
        'sparse_paragraph': 'retrieval_results_ru_sparse_paragraph.jsonl',
        'dense_sentence': 'retrieval_results_ru_dense_sentence.jsonl',
        'dense_paragraph': 'retrieval_results_ru_dense_paragraph.jsonl'
    }
    
    # Get the project root directory (parent of debug/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Dictionary to store all results grouped by query_id
    combined_data = defaultdict(dict)
    
    # Read each file and organize by query_id
    for result_type, filename in input_files.items():
        file_path = project_root / filename
        
        if not file_path.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue
            
        print(f"Reading {filename}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line.strip())
                query_id = entry['query_id']
                # Store the results for this query_id and result_type
                combined_data[query_id][result_type] = entry.get('results', [])
    
    # Write the combined results to a new JSONL file
    output_file = project_root / 'retrieval_results_ru_combined.jsonl'
    print(f"\nWriting combined results to {output_file}...")
    
    # Sort query_ids for consistent output
    sorted_query_ids = sorted(combined_data.keys(), key=lambda x: int(x.split('-')[1]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for query_id in sorted_query_ids:
            combined_entry = {
                'query_id': query_id,
                'retrieval_results': combined_data[query_id]
            }
            f.write(json.dumps(combined_entry, ensure_ascii=False) + '\n')
    
    print(f"\nSuccess! Combined {len(combined_data)} queries.")
    print(f"Output saved to: {output_file}")
    
    # Print some statistics
    print("\nStatistics:")
    for result_type in input_files.keys():
        count = sum(1 for data in combined_data.values() if result_type in data)
        print(f"  {result_type}: {count} queries")

if __name__ == '__main__':
    combine_retrieval_results()
