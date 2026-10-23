# Retrieval pipeline

Run these commands from anywhere. Paths are resolved from the repository root.

## Inputs to place first

Retrieval does not build `ready_chunks.json`. Put one file here for each chunk size:

```text
heavy_artifacts/retrieval/<chunk_size>/ready_chunks.json
```

It is a JSON object from article id to a list of chunk strings:

```json
{
  "12345": ["First chunk.", "Second chunk."]
}
```

Both retrieval scripts read that file. For metrics, `<chunk_size>` has to be `sentences`, `sw_2`, `sw_3`, `sw_4`, `sw_5`, `sw_6`, or `para`.

A ready dense index is optional. Without it, retrieval embeds the missing strings. To skip that, place the index the same way step 1 writes it:

```text
data/vector_store/<model>/<chunk_size>/corpus/embeddings.npy
data/vector_store/<model>/<chunk_size>/corpus/metadata.json
data/vector_store/<model>/<chunk_size>/queries/embeddings.npy
data/vector_store/<model>/<chunk_size>/queries/metadata.json
```

`embeddings.npy` is a float32 matrix, one L2-normalized row per text. `metadata.json` is a JSON list of the raw strings in that same order, without the model prefix. Corpus strings have to match the chunks in `ready_chunks.json`. Query strings have to match the query texts. `profile.json` in those folders is only a record of how the index was built.

The parallel RRF runs load a chunk-level BM25 index from `data/vector_store/bm25/<chunk_size>/` (`index.pkl`, `texts.pkl`, `owners.pkl`, and the bm25s files beside them). `--config dense` does not use it. `--reindex_bm25` rebuilds it from `ready_chunks.json`.

The cascade loads an article-level BM25 index from `data/vector_store/bm25/articles/` and does not rebuild it. Place that directory before `retrieval_2_stage_rrf.py`.

Query and article texts are downloaded, not copied into the repo. Both retrieval scripts use `kaengreg/wikifacts-articles`, configs `corpus` and `queries`. The parallel script can take a local directory instead, via `--local_dataset_path`, containing `corpus.jsonl` and `queries.jsonl`. Each line needs `_id` and `text`. A relative path is resolved from the current directory.

Scoring downloads the chunk dataset named by `--folder`: `kaengreg/wikifacts-sents` for `sentences`, `kaengreg/wikifacts-window_2` for `sw_2`, and the same pattern through `window_6` and `para`, plus the matching `...-qrels` repo.

## Outputs

| Stage | Location |
| --- | --- |
| Dense index | `data/vector_store/<model>/<chunk_size>/corpus/` and `.../queries/` (`embeddings.npy`, `metadata.json`, `profile.json`) |
| Parallel retrieval | `heavy_artifacts/retrieval/<chunk_size>/<model>/dense/results.jsonl`, `.../rrf-k0/results.jsonl`, `.../rrf-k60/results.jsonl` |
| Parallel BM25 index | `data/vector_store/bm25/<chunk_size>/` |
| Cascade retrieval | `heavy_artifacts/retrieval/<chunk_size>/<model>/<penalty>/<k>_<article_weight>_<sentence_weight>.jsonl` |
| Cascade BM25 index | `data/vector_store/bm25/articles/` |
| Score JSON | `heavy_artifacts/evaluation/<chunk_size>/`, same relative path as the jsonl file, with a `.json` suffix |
| All-files spreadsheet | `rusbeir_eval/reports/<chunk_size>.xlsx` |
| One-file metrics | `<results_stem>_metrics.txt` next to the score JSON |

Both retrieval scripts read chunks from `heavy_artifacts/retrieval/<chunk_size>/ready_chunks.json` and vectors from the dense index above.

## Giga and Qwen through vLLM

`giga-3b` (`ai-sage/Giga-Embeddings-instruct-3B-0826`) and `qwen3-4b` (`Qwen/Qwen3-Embedding-4B`) load vLLM inside the indexing process, and again inside retrieval only when a chunk or query is missing from the cache. The call is the offline `vllm.LLM` embedder (`runner="pooling"`, `convert="embed"`). A complete cache does not load the model.

Use a Python environment that has vLLM installed. vLLM picks the GPU from `CUDA_VISIBLE_DEVICES`; `--device` does not apply to this backend.

```bash
CUDA_VISIBLE_DEVICES=0 python retrieval_pipeline/create_dense_index.py \
  --model giga-3b \
  --output-dir data/vector_store/giga-3b/sentences

CUDA_VISIBLE_DEVICES=0 python retrieval_pipeline/create_dense_index.py \
  --model qwen3-4b \
  --output-dir data/vector_store/qwen3-4b/sentences
```

Retrieval uses the same environment when it has to embed a cache miss:

```bash
CUDA_VISIBLE_DEVICES=0 python retrieval_pipeline/retrieval_bm25_and_dense_rrf.py \
  --model giga-3b --config rrf --rrf-k 60
```

`--backend transformers` runs these two models without vLLM. E5, FRIDA, and Arctic stay on Transformers. FRIDA cannot use the vLLM backend.

## 1. Build a dense index

```bash
python retrieval_pipeline/create_dense_index.py \
  --model e5 \
  --output-dir data/vector_store/e5/sentences
```

`--model` is `e5`, `frida`, `arctic-l-v2`, `giga-3b`, or `qwen3-4b`.
The last directory of `--output-dir` is the chunk size used below (`sentences`, `sw_4`, `para`, and so on).

Writes `data/vector_store/e5/sentences/corpus/` and `data/vector_store/e5/sentences/queries/`.

## 2. Retrieve

Parallel baselines. One command per variant:

```bash
python retrieval_pipeline/retrieval_bm25_and_dense_rrf.py --model e5 --config dense
python retrieval_pipeline/retrieval_bm25_and_dense_rrf.py --model e5 --config rrf --rrf-k 0
python retrieval_pipeline/retrieval_bm25_and_dense_rrf.py --model e5 --config rrf --rrf-k 60
```

`dense` is cosine similarity only. `rrf-k0` is BM25 plus dense with no rank smoothing. `rrf-k60` is the same fusion with rank smoothing.

Writes `heavy_artifacts/retrieval/sentences/e5/dense/results.jsonl` (and `rrf-k0`, `rrf-k60`). The BM25 index is stored at `data/vector_store/bm25/sentences/`.

Cascade: BM25 selects articles, then dense search ranks chunks inside those articles.

```bash
python retrieval_pipeline/retrieval_2_stage_rrf.py --model e5 --chunk_size sentences
```

Writes one jsonl per grid point under `heavy_artifacts/retrieval/sentences/e5/both_penalties/`, named `<k>_<article_weight>_<sentence_weight>.jsonl`. The article BM25 index is stored at `data/vector_store/bm25/articles/`.

## 3. Convert jsonl to score JSON

```bash
python retrieval_pipeline/get_scores_batch.py --folder sentences
```

`--folder` is the chunk size. This reads every retrieval jsonl under `heavy_artifacts/retrieval/sentences/` and writes the same relative path under `heavy_artifacts/evaluation/sentences/` as `{query_id: {corpus_id: score}}`. `find_chunks_ids.py` and `adapt_scores_format.py` are called by this script.

## 4. Calculate metrics

Every score file for that chunk size, saved as `rusbeir_eval/reports/sentences.xlsx`:

```bash
python retrieval_pipeline/eval_retrieval.py --folder sentences
```

One file, printed and saved as `results_metrics.txt` beside the JSON:

```bash
python retrieval_pipeline/eval_retrieval_single.py \
  --folder sentences \
  --results heavy_artifacts/evaluation/sentences/e5/rrf-k60/results.json
```

That command writes `heavy_artifacts/evaluation/sentences/e5/rrf-k60/results_metrics.txt`.

`--folder` must match the chunk type. `sentences` uses the sentence qrels. `para` uses the paragraph qrels.
