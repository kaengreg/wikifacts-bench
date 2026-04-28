import json
import argparse


def process_file(input_file, output_file):
    """Adapts scores format from JSONL to JSON."""
    # Dictionary to store the extracted data
    adapted_scores = {}

    # Process each line in the JSONL file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the JSON object from each line
            entry = json.loads(line.strip())
            
            # Extract query_id
            query_id = entry['query_id']
            
            # Initialize dictionary for this query if not exists
            if query_id not in adapted_scores:
                adapted_scores[query_id] = {}
            
            # Counter for <UNDEFINED> IDs in this query
            undefined_counter = 0
            
            # Extract id and similarity for each result
            for result in entry['results']:
                result_id = result['id']
                
                # Handle <UNDEFINED> IDs by adding a suffix
                if result_id == '<UNDEFINED>':
                    result_id = f'<UNDEFINED_{undefined_counter}>'
                    undefined_counter += 1
                
                similarity = result['rrf_score'] * 100  # Multiply by 100
                
                # Store the similarity score
                adapted_scores[query_id][result_id] = similarity

    # Save the adapted scores to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(adapted_scores, f, ensure_ascii=False, indent=4)

    print(f"Successfully processed {len(adapted_scores)} queries")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Adapt scores format from JSONL to JSON.")
    parser.add_argument("--input_file", required=True, help="Path to input .jsonl file.")
    parser.add_argument("--output_file", required=True, help="Path to output .json file.")
    args = parser.parse_args()
    process_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
