import sys
sys.path.append("..")

import json
import argparse

from tqdm import tqdm

from data_loader import load_facts
from retrieval import RelevantRetriever


def retrieve_from_corpus(
    corpus_name='kaengreg/wikifacts-bench', 
    lang='ru',
    mode='sparse',
    splitter='sentence',
):
    corpus_dataset = load_facts(corpus_name, split=f"{lang}_corpus")
    queries_dataset = load_facts(corpus_name, split=f"{lang}_queries")

    # For testing
    # corpus_dataset = dict(list(corpus_dataset.items())[:10])
    # queries_dataset = dict(list(queries_dataset.items())[:10])

    if mode == 'sparse':
        retriever = RelevantRetriever(mode='sparse', splitter=splitter, extra_kwargs={'lang': lang})
    elif mode == 'dense':
        retriever = RelevantRetriever(mode='dense', splitter=splitter)
    else:
        raise ValueError("mode must be 'sparse' or 'dense'")

    retrieval_results = {}

    for qid, qrec in tqdm(queries_dataset.items(), desc="Retrieving relevant fragments from corpus"):
        query_text = qrec.get('text', '')
        linked = qrec.get('linked articles', [])

        # Build per-query document index from linked articles
        article_texts = {}
        for aid in linked:
            art = corpus_dataset.get(aid)
            if not art:
                continue
            article_texts[aid] = art.get('text', '')

        top = retriever.retrieve(query_text, article_texts, top_k=5)

        items = []
        for item in top:
            aid = item['article_id']
            items.append({
                'text': item['text'],
                'similarity': float(item['score']),
                'article_id': aid,
            })

        retrieval_results[qid] = items

    return retrieval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve relevant fragments for queries against a per-query corpus.")
    parser.add_argument("--corpus_name", type=str, default="kaengreg/wikifacts-bench", help="Hugging Face dataset name")
    parser.add_argument("--lang", type=str, default="ru", help="Language code, e.g., 'ru', 'en', ...")
    parser.add_argument("--mode", type=str, choices=["sparse", "dense"], default="sparse", help="Retrieval mode")
    parser.add_argument("--splitter", type=str, choices=["sentence", "paragraph"], default="sentence", help="Fragment granularity")
    args = parser.parse_args()

    corpus_name = args.corpus_name
    lang = args.lang
    mode = args.mode
    splitter = args.splitter

    retrieval_results = retrieve_from_corpus(corpus_name, lang, mode, splitter)

    out_path = f'retrieval_results_{lang}_{mode}_{splitter}.jsonl'
    with open(out_path, 'w') as f:
        for qid, items in retrieval_results.items():
            rec = {
                'query_id': qid,
                'results': items,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote results to {out_path}")
