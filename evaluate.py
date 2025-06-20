import argparse
from data_loader import load_facts
from rag_client import RagClient
from typing import List


def parse_response(rsp: str) -> str:
    text = rsp.lower().strip()
    if 'yes' in text:
        return 'yes'
    if 'no' in text:
        return 'no'
    return 'i don\'t know'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kaengreg/wikifacts-bench")
    parser.add_argument('--model', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--api_url', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--max_attempts', type=int, default=3)
    args = parser.parse_args()

    print("loading_facts")
    ds_dict = load_facts(args.dataset, args.split)
    
    facts = [ds_dict[_id]['text'] for _id  in ds_dict.keys()][:1]
    print(facts)
    labels = []  
    client = RagClient(model_name=args.model, api_url=args.api_url, api_key=args.api_key, max_attempts=args.max_attempts)

    preds = []
    for fact in facts:
        raw = client.call_llm(fact)
        print(raw)
        if raw is None:
            preds.append('i don\'t know')
        else:
            preds.append(parse_response(raw))

    print('Предсказания:')
    for ex, p in zip(ds_dict, preds):
        print(f"{ex['qid']}: {p}")


if __name__ == "__main__":
    main()