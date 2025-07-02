import json
import argparse
import dateparser
import re
import unicodedata
import requests
from tqdm import tqdm 
from collections import defaultdict
from urllib.parse import urljoin, unquote
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='all_facts.json', help="Raw input file path")
    parser.add_argument("--output_corpus", default="corpus.jsonl", help="Output corpus JSONL path")
    parser.add_argument("--output_queries", default="queries.jsonl", help="Output queries JSONL path")
    return parser.parse_args()


def preprocess_text(text):
    exclude_chars = "йё"
    text = re.sub(r'а́', 'а', text)
    text = re.sub(r"==\s*(.*?)\s*==\s*", r"\1 ", text)
    text = text.replace('\xa0', ' ').strip()
    text = unicodedata.normalize('NFC', text)
    result = []
    for char in text:
        if char in exclude_chars:
            result.append(char)
        else:
            char_base = unicodedata.normalize('NFD', char)
            char_without_diacritics = ''.join(
                c for c in char_base if not unicodedata.combining(c)
            )
            result.append(char_without_diacritics)
    
    return ''.join(result)

def get_wikipedia_article(link_url):
    m = re.match(r"https?://([a-z]{2})\.wikipedia\.org/wiki/(.+)", unquote(link_url))
    if not m:
        return ""
    lang, title = m.groups()
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "format": "json",
        "titles": title
    }
    response = requests.get(api_url, params=params)
    data = response.json()
    pages = data.get('query', {}).get('pages', {})
    page = next(iter(pages.values()))
    article_text = page.get('extract', '')
    return preprocess_text(article_text)


def extract_article_title(url):
    parts = url.split('/wiki/')
    return parts[1] if len(parts) > 1 else None



def main(args):
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    facts = []
    for year, months in raw_data.items():
        for month, items in months.items():
            for item in items:
                dt_month = dateparser.parse(f"{month} {year}")
                fact_date = dt_month.strftime("%Y-%m") if dt_month else None
                facts.append({
                    'section': item.get('section'),
                    'text': item.get('text'),
                    'links': item.get('links', []),
                    'relevant_links': item.get('relevant_links', []),
                    'fact_date': fact_date
                })

    processed_links = {}
    corpus_entries = {}
    cid_counter = 0

    if os.path.exists(args.output_corpus):
        with open(args.output_corpus, 'r', encoding='utf-8') as cp_file:
            for line in cp_file:
                try:
                    entry = json.loads(line)
                    url = entry['metadata']['url']
                    cid = entry['id']
                    processed_links[url] = cid
                    corpus_entries[cid] = entry
                    num = int(cid.split('-', 1)[1])
                    cid_counter = max(cid_counter, num + 1)
                except Exception:
                    continue

    cp_out = open(args.output_corpus, 'a', encoding='utf-8')

    for fact in tqdm(facts, desc="Processing facts"):
        for url in fact['links'] + fact['relevant_links']:
            if url and url not in processed_links:
                article_text = get_wikipedia_article(url)
                abstract = article_text.split('\n\n')[0]
                cid = f"c-{cid_counter}"
                cid_counter += 1
                processed_links[url] = cid
                corpus_entries[cid] = {
                    'id': cid,
                    'text': article_text,
                    'abstract': abstract,
                    'metadata': {'url': unquote(url)}
                }

                cp_out.write(json.dumps(corpus_entries[cid], ensure_ascii=False) + '\n')
                cp_out.flush()

    cp_out.close()

    processed_queries = set()
    if os.path.exists(args.output_queries):
        with open(args.output_queries, 'r', encoding='utf-8') as qp_file:
            for line in qp_file:
                try:
                    entry = json.loads(line)
                    processed_queries.add(entry['id'])
                except Exception:
                    continue

    qp_out = open(args.output_queries, 'a', encoding='utf-8')

    queries_entries = []
    for idx, fact in enumerate(tqdm(facts, desc="Building queries")):
        qid = f"q-{idx}"
        if qid in processed_queries:
            continue
        linked_cids = [processed_links[url] for url in fact['links'] if url in processed_links]
        relevant_cids = [processed_links[url] for url in fact['relevant_links'] if url in processed_links]

        article_titles = [
            extract_article_title(url)
            for url in fact['links'] + fact['relevant_links']
            if url in processed_links
        ]
        title_words = set()
        for title in article_titles:
            decoded_title = unquote(title)
            title_text = decoded_title.replace('_', ' ')
            title_words.update(re.findall(r'\w+', title_text.lower(), flags=re.UNICODE))

        fact_words = set(re.findall(r'\w+', fact['text'].lower(), flags=re.UNICODE))
        keywords = sorted(title_words - fact_words)

        queries_entries.append({
            'id': qid,
            'text': fact['text'],
            'linked articles': linked_cids,
            'relevant articles': relevant_cids,
            'keywords': keywords,
            'metadata': {'fact_date': fact['fact_date']}
        })
        qp_out.write(json.dumps(queries_entries[-1], ensure_ascii=False) + '\n')
        qp_out.flush()
    qp_out.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
