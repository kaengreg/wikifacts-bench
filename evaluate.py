import os
import json
import argparse
from rag_client import FactOnlyClient, LinkedAbstractClient, RelevantAbstractClient
from data_loader import load_queries, load_corpus
from collections import Counter
from tqdm import tqdm
from retrieval import RelevantRetriever
import torch
import re  
import spacy
import spacy.cli
from spacy.language import Language
import pymorphy3

class Pymorphy3Lemmatizer:
    """SpaCy pipeline component for lemmatization using PyMorphy3."""
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
    def __call__(self, doc):
        for token in doc:
            if token.is_alpha:
                token.lemma_ = self.morph.parse(token.text)[0].normal_form
        return doc

@Language.factory("pymorphy_lemmatizer")
def create_pymorphy_lemmatizer(nlp, name):
    return Pymorphy3Lemmatizer()


def read_checkpoint(path: str):
    """Reads checkpoint file and load already processed facts"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: checkpoint at {path} is corrupted.")
    return {}


def resolve_context(article_ids, corpus):
    """Returns abstracts from corpus based on list of article IDs."""
    contexts = []
    for aid in article_ids:
        if aid in corpus:
            contexts.append(corpus[aid]['abstract'])
        else:
            print(f"WARNING: Article ID '{aid}' not found in corpus.")
    return contexts
    
def get_rag_client(mode, model_name, api_url, api_key,  failed_facts_path, allow_idk=True):
    """Selects RAG-client mode"""
    base_kwargs = dict(model_name=model_name, api_url=api_url, api_key=api_key, allow_idk=allow_idk, failed_facts_path=failed_facts_path)
    if mode == "fact":
        return FactOnlyClient(**base_kwargs)
    elif mode == "linked":
        return LinkedAbstractClient(**base_kwargs)
    elif mode == "relevant":
        return RelevantAbstractClient(**base_kwargs)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kaengreg/wikifacts-bench')
    parser.add_argument('--lang', type=str, default='ru')
    parser.add_argument('--model', type=str, default='llama3-70b')
    parser.add_argument('--api_url', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--mode', choices=['fact', 'linked', 'relevant'], default='fact')
    parser.add_argument('--allow_idk', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.json')
    parser.add_argument('--outputs', type=str, default='outputs.jsonl')
    parser.add_argument('--results', type=str, default='final_results.json')
    parser.add_argument('--failed_facts', type=str, default='failed_facts.jsonl')
    parser.add_argument('--use_fragment_retriever', action='store_true')
    parser.add_argument('--retriever_model', type=str, default='')
    parser.add_argument('--retriever_top_k', type=int, default=5)
    parser.add_argument('--retriever_splitter', choices=['sentence', 'paragraph'], default='sentence')
    parser.add_argument('--retriever_pooling', choices=['mean', 'cls'], default='mean')
    args = parser.parse_args()

    for path in [args.checkpoint, args.outputs, args.results]:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    queries = dict(list(load_queries(args.dataset, f"{args.lang}_queries").items())[:5])
    corpus = load_corpus(args.dataset, f"{args.lang}_corpus")

    if args.use_fragment_retriever:
        retriever = RelevantRetriever(
            model_name=args.retriever_model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pooling=args.retriever_pooling,
            splitter=args.retriever_splitter
        )
        
    client = get_rag_client(args.mode, args.model, args.api_url, args.api_key, args.failed_facts, allow_idk=args.allow_idk)

    predictions  = read_checkpoint(args.checkpoint)
    remaining = [(qid, q) for qid, q in queries.items() if qid not in predictions]

    coverage_scores = []

    model_name = f"{args.lang}_core_news_sm"
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Downloading…")
        spacy.cli.download(model_name)
        try:
            nlp = spacy.load(model_name)
        except OSError:
            nlp = spacy.blank(args.lang)
            nlp.add_pipe('attribute_ruler')
            nlp.add_pipe('lemmatizer', config={'mode':'rule'})

    nlp.initialize()

    if args.lang.lower() in ('ru', 'uk'):
        nlp.add_pipe('pymorphy_lemmatizer')

    print(f"Running client class: {client.__class__.__name__}")
    for qid, record in tqdm(remaining, desc="Processing facts", total=len(remaining)):
        fact = record['text']
        if args.mode.strip() == "fact":
            prompt, resp_str = client.call_llm(fact)
        elif args.mode.strip() == "linked":
            abstracts = resolve_context(record.get("linked articles", []), corpus)
            prompt, resp_str = client.call_llm(fact, abstracts)
        elif args.mode.strip() == "relevant":
            if args.use_fragment_retriever:
                article_ids = record.get("relevant articles", [])
                all_text = " ".join([corpus[aid]['text'] for aid in article_ids if aid in corpus])
                abstracts = retriever.retrieve(fact, all_text, top_k=args.retriever_top_k)
            else:
                abstracts = resolve_context(record.get("relevant articles", []), corpus)
            prompt, resp_str = client.call_llm(fact, abstracts)
        else:
            raise ValueError("Invalid mode")

        if isinstance(resp_str, dict):
            resp_json = resp_str
        else:
            resp_json = json.loads(resp_str)

        answer = resp_json['answer']
        reasoning = resp_json.get('reasoning', '')

        predictions[qid] = {"answer": answer, "reasoning": reasoning}

        raw_keywords = record.get('keywords', [])

        if raw_keywords: 
            norm_keywords = set()
            for kw in raw_keywords:
                for token in nlp(kw):
                    if token.is_alpha:
                        norm_keywords.add(token.lemma_.lower())

            reason_doc = nlp(reasoning or "")
            reason_words = set(tok.lemma_.lower() for tok in reason_doc if tok.is_alpha)

            matched = norm_keywords & reason_words
            coverage = len(matched) / len(norm_keywords) if norm_keywords else 0.0
            coverage_scores.append(coverage)

        with open(args.checkpoint, 'w', encoding='utf-8') as fcp:
            json.dump(predictions, fcp, ensure_ascii=False, indent=2)
        with open(args.outputs, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps({"prompt": prompt, "prediction": answer, "reasoning": reasoning, 'output': resp_str}, ensure_ascii=False) + '\n')

    all_preds = []
    
    for qid in queries:
        rec = predictions[qid]
        if isinstance(rec, dict):
            all_preds.append(rec["answer"])
        else:
            all_preds.append(rec)

    stats = Counter(all_preds)
    tp = stats['yes']
    fn = sum(1 for p in all_preds if p == 'no')
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = sum(p == 'yes' for p in all_preds) / len(all_preds)
    idk_ratio = stats.get("idk", 0) / len(all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IDK ratio: {idk_ratio:.4f}")
    print(f"Keywords coverage: {sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0}")
    print("Stats:", dict(stats))
    print(f"Mean coverage: {coverage_scores}")

    with open(args.results, 'w', encoding='utf-8') as fout:
        json.dump({
            "model": args.model,
            "accuracy": accuracy,
            "recall (without idk)": recall,
            "idk_ratio": idk_ratio,
            "mean_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0,
            "stats": dict(stats)
        }, fout, ensure_ascii=False, indent=2)
 

if __name__ == "__main__":
    main()