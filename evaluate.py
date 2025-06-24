import argparse
from data_loader import load_facts
from rag_client import RagClient
from typing import List
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm 
import json
import os


def read_checkpoint(chekpoint_path: str): 
    if os.path.exists(chekpoint_path):
        with open(chekpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kaengreg/wikifacts-bench")
    parser.add_argument('--model', type=str, default="llama3-70b")
    parser.add_argument('--split', type=str, default='rus_queries')
    parser.add_argument('--api_url', type=str, default='http://89.169.128.106:6266/v1')
    parser.add_argument('--api_key', type=str, default='874c364705747e7ab314ceba89c2029c9a72ab2154664c470eb4ce18c2f0acb0')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.jsonl')
    parser.add_argument('--outputs', type=str, default='outputs.jsonl')
    parser.add_argument('--out_file', type=str, default='results.json')
    parser.add_argument('--max_attempts', type=int, default=3)

    args = parser.parse_args()

    ds_dict = load_facts(args.dataset, args.split)
    
    facts = [ds_dict[_id]['text'] for _id  in ds_dict.keys()]
    
    predictions = {}
    if os.path.exists(args.checkpoint_file):
        predictions = read_checkpoint(args.checkpoint_file)

    remaining_facts = [(i, fact) for i, fact in enumerate(facts) if str(i) not in predictions]

    client = RagClient(model_name=args.model, api_url=args.api_url, api_key=args.api_key, max_attempts=args.max_attempts)

    
    for i, fact in tqdm(remaining_facts, desc="Processing facts", total=len(remaining_facts)):
        raw = client.call_llm(fact, no_think=True)
        predictions[str(i)] = raw

        with open(args.checkpoint_file, 'w', encoding='utf-8') as fc:
            json.dumps(predictions, fc, ensure_ascii=False, indent=2)
        with open(args.outputs, 'w', encoding='utf-8') as fo:
            fo.write(json.dumps({'fact': fact, 'prediction': raw}, ensure_ascii=False) + '\n')

    preds = [predictions[str(i)] for i in range(len(facts))]

    true_labels = ['yes'] * len(preds)

    precision = sum(pred == 'yes' for pred in preds) / len(preds)
    tp = sum(1 for true, pred in zip(true_labels, preds) if true == 'yes' and pred == 'yes')
    fn = sum(1 for true, pred in zip(true_labels, preds) if true == 'yes' and pred != 'yes')
    recall = tp / tp + fn if (tp + fn) > 0.0 else 0.0

    idk_ratio = preds.count("i don't know") / len(preds)

    stats = dict.fromkeys(['yes', 'no', "i don't know"], 0)
    for pred in preds:
        match pred:
            case 'yes':
                stats['yes'] += 1
            case 'no':
                stats['no'] += 1
            case "i don't know":
                stats["i don't know"] += 1
        

    print(f"Results for model {args.model}: \n precision: {precision:.3f} \n Recall: {recall:.3f} \n Idk-ratio: {idk_ratio:.3f}\n")
    print(f"Answer stats: ")
    for key, value in stats.items():
        print(f" {key}: {value}")

    with open(args.out_file, 'w', encoding='utf-8') as f:
        json.dump({'model': args.model, 
                   'precision': precision,
                   'recall': recall,
                   'idk_ration': idk_ratio,
                   'stats': stats}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()