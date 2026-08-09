import os
import argparse
import json
import sys
from pathlib import Path

# Add current directory to sys.path to allow importing fellow scripts
sys.path.append(str(Path(__file__).parent))

from find_chunks_ids import (
    load_sents_text_to_id,
    load_sents_qid_to_rels,
    process_file as find_chunks_process
)
from adapt_scores_format import process_file as adapt_scores_process


FOLDER_TO_DATASET = {
    'sentences': 'sents',
    'sw_2': 'window_2',
    'sw_3': 'window_3',
    'sw_4': 'window_4',
    'sw_5': 'window_5',
    'sw_6': 'window_6',
    'para': 'para',
}


def main():
    parser = argparse.ArgumentParser(description="Batch process retrieval scores.")
    parser.add_argument("--folder", required=True, help="Folder inside heavy_artifacts/retrieval to process (e.g., 'sentences').")
    args = parser.parse_args()

    base_input_dir = Path("heavy_artifacts/retrieval") / args.folder
    base_output_dir = Path("heavy_artifacts/evaluation") / args.folder

    dataset_name = FOLDER_TO_DATASET[args.folder]

    # Step 1: Load HuggingFace datasets once
    hf_dataset_name = f"kaengreg/wikifacts-{dataset_name}"
    qrels_name = f"kaengreg/wikifacts-{dataset_name}-qrels"

    print(f"Loading {hf_dataset_name} and creating text_to_id mapping...")
    text_to_id = load_sents_text_to_id(hf_dataset_name, split="corpus")

    print(f"Loading {qrels_name} and creating qid_to_rels mapping...")
    qid_to_rels = load_sents_qid_to_rels(qrels_name, split="qrels")

    if not base_input_dir.exists():
        print(f"Error: Input directory {base_input_dir} does not exist.")
        return

    # Find all .jsonl files recursively
    jsonl_files = list(base_input_dir.rglob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No .jsonl files found in {base_input_dir}")
        return

    print(f"Found {len(jsonl_files)} files to process.")

    for input_file in jsonl_files:
        # Determine relative path to maintain structure
        rel_path = input_file.relative_to(base_input_dir)
        output_file = base_output_dir / rel_path.with_suffix(".json")
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Temporary file for the intermediate step
        temp_file = output_file.with_suffix(".temp.jsonl")

        print(f"\nProcessing {input_file}...")

        # Step 2: Run find_chunks_ids logic
        print(f"Running find_chunks_ids logic...")
        try:
            find_chunks_process(input_file, temp_file, text_to_id, qid_to_rels)
        except Exception as e:
            print(f"Error processing {input_file} in find_chunks_ids: {e}")
            continue

        # Step 3: Run adapt_scores_format logic
        print(f"Running adapt_scores_format logic...")
        try:
            adapt_scores_process(temp_file, output_file)
        except Exception as e:
            print(f"Error processing {temp_file} in adapt_scores_format: {e}")
            if temp_file.exists():
                os.remove(temp_file)
            continue

        # Step 4: Cleanup temporary file
        if temp_file.exists():
            os.remove(temp_file)

        print(f"Finished processing. Output saved to: {output_file}")

    print("\nAll files processed successfully.")


if __name__ == "__main__":
    main()
