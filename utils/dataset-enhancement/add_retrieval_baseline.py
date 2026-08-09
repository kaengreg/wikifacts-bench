import json
from pathlib import Path


def update_queries_with_retrieval_results():
    """
    Updates queries_ru.jsonl with retrieval results from the combined file.
    Adds retrieval_results under the metadata attribute for each query.
    """
    
    # Get the project root directory (parent of debug/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Define file paths
    queries_file = project_root / 'queries_ru.jsonl'
    retrieval_file = project_root / 'retrieval_results_ru_combined.jsonl'
    output_file = project_root / 'queries_ru_updated.jsonl'
    
    # Check if files exist
    if not queries_file.exists():
        print(f"Error: {queries_file} not found!")
        return
    
    if not retrieval_file.exists():
        print(f"Error: {retrieval_file} not found!")
        return
    
    # Load retrieval results into a dictionary keyed by query_id
    print(f"Loading retrieval results from {retrieval_file}...")
    retrieval_data = {}
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            query_id = entry['query_id']
            retrieval_data[query_id] = entry['retrieval_results']
    
    print(f"Loaded retrieval results for {len(retrieval_data)} queries.")
    
    # Update queries with retrieval results
    print(f"Updating queries from {queries_file}...")
    updated_count = 0
    missing_count = 0
    
    with open(queries_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            query_entry = json.loads(line.strip())
            query_id = query_entry['id']
            
            # Add retrieval results to metadata if available
            if query_id in retrieval_data:
                # Ensure metadata exists
                if 'metadata' not in query_entry:
                    query_entry['metadata'] = {}
                
                # Add retrieval results to metadata
                query_entry['metadata']['retrieval_baseline'] = retrieval_data[query_id]
                updated_count += 1
            else:
                missing_count += 1
                print(f"Warning: No retrieval results found for {query_id}")
            
            # Write updated entry
            f_out.write(json.dumps(query_entry, ensure_ascii=False) + '\n')
    
    print(f"\nSuccess!")
    print(f"  Updated queries: {updated_count}")
    print(f"  Missing retrieval results: {missing_count}")
    print(f"  Output saved to: {output_file}")

if __name__ == '__main__':
    update_queries_with_retrieval_results()
