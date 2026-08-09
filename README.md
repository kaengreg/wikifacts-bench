
# WikiFactsBench

**WikiFactsBench** is a live, multilingual benchmark designed specifically for evaluating Retrieval-Augmented Generation (RAG) systems.

This benchmark serves as an evaluation framework for large language models (LLMs) within RAG systems. The evaluation relies on a curated dataset derived from Wikipedia’s “Did you know…” sections. These datasets are specifically constructed to assess the knowledge retrieval and reasoning capabilities of LLMs. The resource currently supports multiple languages and is continuously updated to reflect the evolving content of Wikipedia.

### Features
-	**Multilingual** evaluation across 10 supported languages
-	Four evaluation modes:
    -	fact-only
    -	linked abstracts
    -	relevant abstracts
    -	relevant fragments (retrieved with a retriever model)
-	OpenAI-compatible client - use your own API base URL + key
-	Flexible prompting - optional few-shot examples and support for “I don’t know” answers

### 🌍 Supported Languages

The benchmark datasets are available on [HuggingFace](https://huggingface.co/datasets/kaengreg/wikifacts-bench).
Currently supported languages:
-	🇷🇺 Russian (ru)
-	🇬🇧 English (en)
-	🇩🇪 German (de)
-	🇫🇷 French (fr)
-	🇵🇹 Portuguese (pt)
-	🇨🇳 Chinese (zh)
-	🇺🇦 Ukrainian (uk)
-	🇳🇱 Dutch (nl)
-	🇸🇪 Swedish (sv)
-	🇻🇳 Vietnamese (vi)


### 📦 Installation
```
git clone https://github.com/kaengreg/wikifacts-bench.git
cd wikifacts-bench
```
Requirements are listed in requirements.txt. Use ```Python 3.11+``` for best compatibility.
```
pip install -r requirements.txt
```
### Running the Benchmark

evaluate.py is the main entry point. It evaluates an LLM on a chosen dataset/language and saves outputs and metrics.

#### Supported modes

Select a mode by passing one of the following options to evaluate.py:
-	```--mode fact``` - ask the model the fact without any attached context
-	```--mode linked``` - ask with article abstracts referenced in the fact
-	```--mode relevant``` - ask with relevant fragments (abstracts from relevant articles or retrieved fragments)

##### Additional flags
-	Allow “I don’t know” answers: --allow_idk
-	Enable few-shot prompting: --use_few_shots
-	Translate base prompts from English to the target language: --translate_prompts

### Examples

Replace the URL/key with your own OpenAI-compatible endpoint.

1) **Fact-only** (no retrieval/context)
```
python evaluate.py \
  --dataset kaengreg/wikifacts-bench \
  --lang ru \
  --mode fact \
  --model llama3-70b \
  --api_url http://localhost:8000/v1 \
  --api_key YOUR_API_KEY
```
2) **With linked abstracts**
```
python evaluate.py \
  --dataset kaengreg/wikifacts-bench \
  --lang ru \
  --mode linked \
  --model llama3-70b \
  --api_url http://localhost:8000/v1 \
  --api_key YOUR_API_KEY
```
3) **With relevant fragments + retriever settings**
```
python evaluate.py \
  --dataset kaengreg/wikifacts-bench \
  --lang ru \
  --mode relevant \
  --model llama3-70b \
  --api_url http://localhost:8000/v1 \
  --api_key YOUR_API_KEY \
  --retriever_model intfloat/multilingual-e5-large \
  --retriever_top_k 8 \
  --retriever_splitter sentence \
  --retriever_pooling mean
```
**Outputs**:
-	Per-question generations → outputs.jsonl
-	Aggregated metrics (accuracy, recall (w/o IDK), IDK ratio, mean coverage, stats) → final_results.json
-	Failures (e.g., JSON parse errors) → failed_facts.jsonl

### Building a Corpus from Wikipedia

You can build your own dataset for a language using the provided utilities.

#### 1) Parse Wikipedia “Did you know…”

```utils/wikifacts-parsing``` provides scripts for parsing Wikipedia “Did you know…” pages. You can use them to create your own corpus or as a foundation for developing parsing scripts for other languages.

The {lang}-wiki_parse.py script fetches the “Did you know…” archive, producing a single JSON file with all facts.
```
python {lang}-wiki_parse.py
```


#### 2) Create corpus + queries

corpus-articles.py converts the parsed JSON into the corpus (corpus.jsonl) and queries (queries.jsonl) used by the benchmark.

##### Examples

Generate both corpus and queries for Russian:
```
python corpus-articles.py \
  --input_file data/ru/all_facts.json \
  --output_corpus data/ru/corpus.jsonl \
  --output_queries data/ru/queries.jsonl \
  --lang ru
```
Generate only the corpus:
```
python corpus-articles.py \
  --input_file data/ru/all_facts.json \
  --output_corpus data/ru/corpus.jsonl \
  --lang ru \
  --corpus_only
```
Generate only the queries:
```
python corpus-articles.py \
  --input_file data/ru/all_facts.json \
  --output_queries data/ru/queries.jsonl \
  --lang ru \
  --queries_only
```
Once produced, point evaluate.py at the dataset you want to use (the default expects kaengreg/wikifacts-bench).


### 📜 License

This project is licensed under the Apache 2.0 License. See LICENSE for details.


Got it 👍 Here’s how you could add a clean “Contact” section at the end of your README:



### 📬 Contacts

Feel free to report any problems or bugs by opening an issue on GitHub or reaching out directly via email:
[Contact the maintainer](mailto:kaengreg@ya.ru?subject=WikiFactsBench%20Bug%20Report&body=Please%20describe%20the%20issue%20here...)
