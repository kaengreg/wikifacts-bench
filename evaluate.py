import argparse
from data_loader import load_facts
from rag_client import RagClient
from typing import List
from sklearn.metrics import accuracy_score, recall_score

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="kaengreg/wikifacts-bench")
    parser.add_argument('--model', type=str, default="llama3-70b")
    parser.add_argument('--split', type=str, default='rus_queries')
    parser.add_argument('--api_url', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_attempts', type=int, default=3)
    args = parser.parse_args()

    ds_dict = load_facts(args.dataset, args.split)
    
    facts = [ds_dict[_id]['text'] for _id  in ds_dict.keys()][:5]
    labels = []  
    client = RagClient(model_name=args.model, api_url=args.api_url, api_key=args.api_key, max_attempts=args.max_attempts)

    preds = []
    for fact in facts:
        raw = client.call_llm(fact, no_think=True)
        preds.append(raw)


    true_labels = ['yes'] * len(preds)
    accuracy = accuracy_score(true_labels, preds)
    recall = recall_score(true_labels, preds, pos_label='yes')

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
        

    print(f"Results for model {args.model}: \n Accuracy: {accuracy:.3f} \n Recall: {recall:.3f} \n Idk-ratio: {idk_ratio:.3f}\n")
    print(f"Answer stats: ")
    for key, value in stats.items():
        print(f" {key}: {value}")

if __name__ == "__main__":
    main()