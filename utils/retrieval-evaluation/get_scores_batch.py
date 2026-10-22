"""Turn retrieval jsonl files into the JSON that the BEIR scorer reads.

Run this from anywhere. Paths are resolved from the repository root.

``--folder`` is the chunk directory (``sentences``, ``sw_4``, ``para``, ...).
Every retrieval ``.jsonl`` under ``heavy_artifacts/retrieval/<folder>/`` is
converted, and the same relative path is written under
``heavy_artifacts/evaluation/<folder>/``.

That covers the old cascade files and the current layout:

    <model>/<penalty>/<k>_<article_weight>_<sentence_weight>.jsonl
    <model>/dense/results.jsonl
    <model>/rrf-k0/results.jsonl
    <model>/rrf-k60/results.jsonl

Each input line needs ``query_id`` and ``results[].text`` plus
``results[].rrf_score``. The output is ``{query_id: {corpus_id: score}}``.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]

FOLDER_TO_DATASET = {
    'sentences': 'sents',
    'sw_2': 'window_2',
    'sw_3': 'window_3',
    'sw_4': 'window_4',
    'sw_5': 'window_5',
    'sw_6': 'window_6',
    'para': 'para',
}


def is_retrieval_jsonl(path: Path) -> bool:
    """A retrieval file's first record has a query id and a result list."""
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False
            return isinstance(entry, dict) and "query_id" in entry and isinstance(entry.get("results"), list)
    return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert retrieval jsonl files to BEIR score JSON. "
            "Walks heavy_artifacts/retrieval/<folder>/ at any depth, "
            "including <model>/dense|rrf-k0|rrf-k60/results.jsonl and cascade grids."
        )
    )
    parser.add_argument(
        "--folder",
        required=True,
        choices=sorted(FOLDER_TO_DATASET),
        help="Chunk directory inside heavy_artifacts/retrieval.",
    )
    args = parser.parse_args()

    base_input_dir = REPO_ROOT / "heavy_artifacts" / "retrieval" / args.folder
    base_output_dir = REPO_ROOT / "heavy_artifacts" / "evaluation" / args.folder

    dataset_name = FOLDER_TO_DATASET[args.folder]

    # Step 1: Load HuggingFace datasets once
    hf_dataset_name = f"kaengreg/wikifacts-{dataset_name}"
    qrels_name = f"kaengreg/wikifacts-{dataset_name}-qrels"

    print(f"Chunk folder: {args.folder}")
    print(f"Corpus: {hf_dataset_name}")
    print(f"Qrels: {qrels_name}")
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

    print(f"Found {len(jsonl_files)} .jsonl files under {base_input_dir}.")

    for input_file in jsonl_files:
        if not is_retrieval_jsonl(input_file):
            print(f"Skipping {input_file}: the first record is not a retrieval result.")
            continue
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
