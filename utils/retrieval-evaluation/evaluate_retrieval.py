# Requires RusBEIR project to be in the same folder

import json

from rusBeIR.beir.datasets.data_loader_hf import HFDataLoader
from rusBeIR.beir.retrieval.evaluation import EvaluateRetrieval


corpus, queries, qrels = HFDataLoader(
    hf_repo="kaengreg/wikifacts-sents", 
    hf_repo_qrels="kaengreg/wikifacts-sents-qrels", 
    streaming=False,
    keep_in_memory=False
).load(split='dev')

with open('adapted_scores_pre_split.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

# print(qrels)
# print("--------------------------------")
# # Debug
# results = {"bwq-2453": results["bwq-2453"]}
# print(results)
# qrels = {"bwq-2453": qrels["bwq-2453"]}
# print("--------------------------------")
# print(qrels)

retriever = EvaluateRetrieval(k_values=[1,3,5,10, 100])

ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")

metrics = {"ndcg": ndcg, "_map": _map, "recall": recall, "precision": precision, "mrr": mrr}

for metric in metrics.keys():
    for it_num, it_val in zip(metrics[metric], metrics[metric].values()):
        print(it_num, it_val )
    print('\n')
