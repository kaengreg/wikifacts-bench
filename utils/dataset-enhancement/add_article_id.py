#!/usr/bin/env python3
"""
Script to add article_id attribute from corpus_articles.jsonl to sentences in corpus_sents.jsonl
"""

import json
from typing import Optional, Dict, List
from tqdm import tqdm


def load_articles(articles_file: str) -> List[Dict]:
    """Load all articles from JSONL file."""
    print(f"Loading articles from {articles_file}...")
    articles = []
    with open(articles_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles.")
    return articles


def find_article_id(sentence_text: str, articles: List[Dict], last_article_cache: Optional[Dict] = None) -> tuple[str, Optional[Dict]]:
    """
    Find the article ID that contains the given sentence text.
    
    Args:
        sentence_text: The sentence text to search for
        articles: List of all article dictionaries
        last_article_cache: The last matched article (for optimization)
    
    Returns:
        Tuple of (article_id, matched_article) where matched_article is the article dict if found, None otherwise
    """
    # First, check the cached article (if provided)
    if last_article_cache is not None:
        if sentence_text in last_article_cache['text']:
            return last_article_cache['_id'], last_article_cache
    
    # If not in cache, search through all articles
    for article in articles:
        if sentence_text in article['text']:
            return article['_id'], article
    
    # No match found
    return "<NO_VALUE>", None


def process_sentences(sentences_file: str, articles_file: str, output_file: str):
    """
    Process all sentences and add article_id attribute.
    
    Args:
        sentences_file: Path to corpus_sents.jsonl
        articles_file: Path to corpus_articles.jsonl
        output_file: Path to output file with article IDs added
    """
    # Load all articles
    articles = load_articles(articles_file)
    
    # Process sentences
    print(f"Processing sentences from {sentences_file}...")
    
    matched_count = 0
    unmatched_count = 0
    cache_hit_count = 0
    last_article_cache = None
    
    with open(sentences_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # Count total lines for progress bar
        print("Counting sentences...")
        total_lines = sum(1 for _ in open(sentences_file, 'r', encoding='utf-8'))
        
        # Reset file pointer
        f_in.seek(0)
        
        for line in tqdm(f_in, total=total_lines, desc="Matching sentences to articles"):
            if not line.strip():
                continue
            
            sentence = json.loads(line)
            sentence_text = sentence['text']
            
            # Track if this was a cache hit
            was_cache_hit = last_article_cache is not None and sentence_text in last_article_cache['text']
            
            # Find the article ID
            article_id, matched_article = find_article_id(sentence_text, articles, last_article_cache)
            
            # Update cache if we found a new article
            if matched_article is not None:
                last_article_cache = matched_article
            
            # Update statistics
            if article_id == "<NO_VALUE>":
                unmatched_count += 1
            else:
                matched_count += 1
                if was_cache_hit:
                    cache_hit_count += 1
            
            # Add article_id to sentence
            sentence['article_id'] = article_id
            
            # Write to output file
            f_out.write(json.dumps(sentence, ensure_ascii=False) + '\n')
    
    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total sentences: {matched_count + unmatched_count}")
    print(f"Matched: {matched_count}")
    print(f"Unmatched: {unmatched_count}")
    print(f"Cache hits: {cache_hit_count} ({100 * cache_hit_count / (matched_count + unmatched_count):.2f}%)")
    print(f"Output written to: {output_file}")


if __name__ == "__main__":
    sentences_file = "corpus_sents.jsonl"
    articles_file = "corpus_articles.jsonl"
    output_file = "corpus_sents_with_article_ids.jsonl"
    
    process_sentences(sentences_file, articles_file, output_file)

