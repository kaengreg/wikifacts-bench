import os
import json
import argparse
from rag_client import FactOnlyClient, LinkedAbstractClient, RelevantAbstractClient
from data_loader import load_queries, load_corpus
from collections import Counter
from tqdm import tqdm


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
    
def get_rag_client(mode, model_name, api_url, api_key, allow_idk=True):
    """Selects RAG-client mode"""
    base_kwargs = dict(model_name=model_name, api_url=api_url, api_key=api_key, allow_idk=allow_idk)
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
    parser.add_argument('--lang', type=str, default='rus')
    parser.add_argument('--model', type=str, default='llama3-70b')
    parser.add_argument('--api_url', type=str, default='')
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--mode', choices=['fact', 'linked', 'relevant'], default='linked')
    parser.add_argument('--allow_idk', action='store_true', default=True)
    parser.add_argument('--checkpoint', type=str, default='checkpoint.json')
    parser.add_argument('--outputs', type=str, default='outputs.jsonl')
    args = parser.parse_args()

    queries = load_queries(args.dataset, f"{args.lang}_queries")
    corpus = load_corpus(args.dataset, f"{args.lang}_corpus")
    client = get_rag_client(args.mode, args.model, args.api_url, args.api_key, allow_idk=args.allow_idk)

    predictions  = read_checkpoint(args.checkpoint)
    remaining = [(qid, q) for qid, q in queries.items() if qid not in predictions]

    print(f"Running client class: {client.__class__.__name__}")
    for qid, record in tqdm(remaining, desc="Processing facts", total=len(remaining)):
        fact = record['text']
        if args.mode.strip() == "fact":
            prompt, pred = client.call_llm(fact)
        elif args.mode.strip() == "linked":
            abstracts = resolve_context(record.get("linked articles", []), corpus)
            prompt, pred = client.call_llm(fact, abstracts)
        elif args.mode.strip() == "relevant":
            abstracts = resolve_context(record.get("relevant articles", []), corpus)
            prompt, pred = client.call_llm(fact, abstracts)
        else:
            raise ValueError("Invalid mode")

        predictions[qid] = pred

        with open(args.checkpoint, 'w', encoding='utf-8') as fcp:
            json.dump(predictions, fcp, ensure_ascii=False, indent=2)
        with open(args.outputs, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps({"prompt": prompt, "prediction": pred}, ensure_ascii=False) + '\n')

    all_preds = [predictions.get(qid, predictions[qid]) for qid in queries]

    stats = Counter(all_preds)
    tp = stats['yes']
    fn = sum(1 for p in all_preds if p == 'no')
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = sum(p == 'yes' for p in all_preds) / len(all_preds)
    idk_ratio = stats.get("i don't know", 0) / len(all_preds)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IDK ratio: {idk_ratio:.4f}")
    print("Stats:", dict(stats))

    with open("final_results.json", 'w', encoding='utf-8') as fout:
        json.dump({
            "model": args.model,
            "accuracy": accuracy,
            "recall (without idk)": recall,
            "idk_ratio": idk_ratio,
            "stats": dict(stats)
        }, fout, ensure_ascii=False, indent=2)
 

if __name__ == "__main__":
    main()