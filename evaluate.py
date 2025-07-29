import os
import json
import argparse
from rag_client import FactOnlyClient, LinkedAbstractClient, RelevantAbstractClient
from data_loader import load_queries, load_corpus
from collections import Counter
from tqdm import tqdm
from retrieval import RelevantRetriever
from lemmatizer import MultilingualLemmatizer
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from deep_translator import GoogleTranslator

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
    
def get_rag_client(mode, model_name, api_url, api_key,  failed_facts_path, translator, use_few_shots, allow_idk=True):
    """Selects RAG-client mode"""
    base_kwargs = dict(model_name=model_name, 
                       api_url=api_url, 
                       api_key=api_key, 
                       failed_facts_path=failed_facts_path, 
                       translator=translator,
                       use_few_shots=use_few_shots,
                       allow_idk=allow_idk)
    
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
    parser.add_argument('--translate_prompts', action='store_true')
    parser.add_argument('--use_few_shots', action='store_true')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.json')
    parser.add_argument('--outputs', type=str, default='outputs.jsonl')
    parser.add_argument('--results', type=str, default='final_results.json')
    parser.add_argument('--failed_facts', type=str, default='failed_facts.jsonl')
    parser.add_argument('--max_threads', type=int, default=10)
    parser.add_argument('--use_fragment_retriever', action='store_true')
    parser.add_argument('--retriever_model', type=str, default='')
    parser.add_argument('--retriever_top_k', type=int, default=5)
    parser.add_argument('--retriever_splitter', choices=['sentence', 'paragraph'], default='sentence')
    parser.add_argument('--retriever_pooling', choices=['mean', 'cls'], default='mean')
    args = parser.parse_args()

    start = time.time()

    translator = None
    if args.translate_prompts:
        translator = GoogleTranslator(source='en', target=args.lang)

    for path in [args.checkpoint, args.outputs, args.results]:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    queries = load_queries(args.dataset, f"{args.lang}_queries")
    corpus = load_corpus(args.dataset, f"{args.lang}_corpus")

    if args.use_fragment_retriever:
        retriever = RelevantRetriever(
            model_name=args.retriever_model,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            pooling=args.retriever_pooling,
            splitter=args.retriever_splitter
        )
        
    client = get_rag_client(args.mode, 
                            args.model,
                            args.api_url, 
                            args.api_key, 
                            args.failed_facts, 
                            allow_idk=args.allow_idk, 
                            translator=translator, 
                            use_few_shots=args.use_few_shots)

    predictions  = read_checkpoint(args.checkpoint)

    remaining = [qid for qid in queries if qid not in predictions]
    if not remaining:
        print("No remaining facts to process. Computing metrics from existing predictions...")

    coverage_scores = []

    lemmatizer = MultilingualLemmatizer(args.lang)

    print(f"Running client class: {client.__class__.__name__}")
    def llm_worker(qid):
        record = queries[qid]
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
            raise ValueError(f"Unsupported mode: {args.mode}")
        return qid, prompt, resp_str

    if remaining:
        executor = ThreadPoolExecutor(max_workers=min(args.max_threads, len(remaining)))
        futures = {executor.submit(llm_worker, qid): qid for qid in remaining}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing facts"):
            qid, prompt, resp_str = fut.result()
            record = queries[qid]
            if resp_str is None:
                predictions[qid] = None
                with open(args.checkpoint, 'w', encoding='utf-8') as fcp:
                    json.dump(predictions, fcp, ensure_ascii=False, indent=2)
                continue
            if isinstance(resp_str, dict):
                resp_json = resp_str
            else:
                resp_json = json.loads(resp_str)
            answer = resp_json['answer'].lower()
            reasoning = resp_json.get('reasoning', '')
            coverage = None
            raw_keywords = record.get('keywords', [])
            if raw_keywords:
                norm_keywords = set()
                for kw in raw_keywords:
                    for token in lemmatizer.nlp(kw):
                        if token.is_alpha:
                            norm_keywords.add(token.lemma_.lower())
                reason_doc = lemmatizer.nlp(reasoning or "")
                reason_words = set(tok.lemma_.lower() for tok in reason_doc if tok.is_alpha)
                matched = norm_keywords & reason_words
                coverage = len(matched) / len(norm_keywords) if norm_keywords else 0.0
                coverage_scores.append(coverage)
            predictions[qid] = {"answer": answer, "reasoning": reasoning, "coverage": coverage}
            with open(args.checkpoint, 'w', encoding='utf-8') as fcp:
                json.dump(predictions, fcp, ensure_ascii=False, indent=2)
            with open(args.outputs, 'a', encoding='utf-8') as fout:
                fout.write(json.dumps({"prompt": prompt, 
                                       "prediction": answer, 
                                       "reasoning": reasoning, 
                                       'output': resp_str}, ensure_ascii=False) + '\n')
        executor.shutdown()
    
    all_preds = []
    for qid in queries:
        rec = predictions[qid]
        if isinstance(rec, dict):
            all_preds.append(rec["answer"])
        else:
            all_preds.append(rec)

    # normalize localized answers to English for metrics
    answer_map = {
        'ru': {'да': 'yes', 'нет': 'no', 'не знаю': 'idk'},
        'en': {'yes': 'yes', 'no': 'no', 'idk': 'idk'},
        'de': {'ja': 'yes', 'nein': 'no', 'weiß nicht': 'idk', 'weiss nicht': 'idk'},
        'fr': {'oui': 'yes', 'non': 'no', 'je ne sais pas': 'idk'},
        'zh': {'是': 'yes', '否': 'no', '不知道': 'idk'},
        'pt': {'sim': 'yes', 'não': 'no', 'nao': 'no', 'não sei': 'idk', 'nao sei': 'idk'}
    }
    map_for_lang = answer_map.get(args.lang, answer_map.get('en', {}))
    
    normalized_preds = []
    for p in all_preds:
        if isinstance(p, str):
            key = p.strip().lower()
        else:
            key = 'idk'  
        normalized_preds.append(map_for_lang.get(key, key))
    all_preds = normalized_preds

    stats = Counter(all_preds)
    tp = stats['yes']
    fn = sum(1 for p in all_preds if p == 'no')
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = sum(p == 'yes' for p in all_preds) / len(all_preds)
    idk_ratio = stats.get("idk", 0) / len(all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IDK ratio: {idk_ratio:.4f}")

    all_coverages = [
        rec.get("coverage", 0.0) for rec in predictions.values()
        if isinstance(rec, dict) and rec.get("coverage") is not None
    ]
    
    mean_coverage = sum(all_coverages) / len(all_coverages) if all_coverages else 0.0
    print(f"Keywords coverage: {mean_coverage:.4f}")
    print("Stats:", dict(stats))

    with open(args.results, 'w', encoding='utf-8') as fout:
        json.dump({
            "model": args.model,
            "accuracy": round(accuracy, 2),
            "recall (without idk)": round(recall, 2),
            "idk_ratio": round(idk_ratio, 2),
            "mean_coverage": round(mean_coverage, 2),
            "stats": dict(stats)
        }, fout, ensure_ascii=False, indent=2)
    
    end = time.time()

    print(f"Overall time: {end - start}")

if __name__ == "__main__":
    main()