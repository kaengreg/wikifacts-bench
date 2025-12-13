#!/usr/bin/env python3
"""
Script to group sentences by article_id from corpus_sents_with_article_ids.jsonl
and output as a JSON file with article IDs as keys and lists of sentence texts as values.
"""

import json
from collections import defaultdict
from tqdm import tqdm


def group_sentences_by_article(input_file: str, output_file: str):
    """
    Group sentences by article_id and save to JSON file.
    
    Args:
        input_file: Path to corpus_sents_with_article_ids.jsonl
        output_file: Path to output JSON file
    """
    print(f"Reading sentences from {input_file}...")
    
    # Dictionary to store grouped sentences
    articles_sentences = defaultdict(list)
    
    # Count total lines for progress bar
    print("Counting sentences...")
    total_lines = sum(1 for line in open(input_file, 'r', encoding='utf-8') if line.strip())
    
    # Read and group sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Grouping sentences by article"):
            if not line.strip():
                continue
            
            sentence = json.loads(line)
            article_id = sentence.get('article_id')
            sentence_text = sentence.get('text', '')
            
            # Add sentence text to the corresponding article
            articles_sentences[article_id].append(sentence_text)
    
    # Convert defaultdict to regular dict for JSON serialization
    articles_sentences = dict(articles_sentences)
    
    # Print statistics
    print(f"\nGrouping complete!")
    print(f"Total unique articles: {len(articles_sentences)}")
    print(f"Total sentences: {sum(len(sents) for sents in articles_sentences.values())}")
    
    if "<NO_VALUE>" in articles_sentences:
        print(f"Sentences without article match: {len(articles_sentences['<NO_VALUE>'])}")
    
    # Save to JSON file
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles_sentences, f, ensure_ascii=False, indent=2)
    
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    input_file = "corpus_sents_with_article_ids.jsonl"
    output_file = "articles_pre_split_sents.json"
    
    group_sentences_by_article(input_file, output_file)

