"""BM25 and dense retrieval, with an optional reciprocal-rank fusion.

Paths are resolved from the repository root, so the working directory does not matter.

``--config dense`` ranks chunks by cosine similarity.
``--config rrf --rrf-k 0`` fuses BM25 and dense as ``1 / rank``.
``--config rrf --rrf-k 60`` fuses them as ``1 / (60 + rank)``.
The default run is ``--model e5 --config rrf --rrf-k 60``.

Dense vectors are read from
``data/vector_store/<model>/<chunk_size>/{corpus,queries}``.
Build that layout with:

    python retrieval_pipeline/create_dense_index.py \\
      --model <profile key> \\
      --output-dir data/vector_store/<profile key>/<chunk_size>

``metadata.json`` has to contain the chunk strings from ``ready_chunks.json``
and the query strings from the queries dataset. A cache hit returns the stored
vector. A miss is embedded with that profile's prefixes.
"""

import importlib.util
import json
import os
import pickle
import sys
from typing import List, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import torch
from datasets import load_dataset

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
from nltk import sent_tokenize

from bm25s import BM25

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
if NOTEBOOKS_DIR not in sys.path:
    sys.path.insert(0, NOTEBOOKS_DIR)

from lemmatizer import MultilingualLemmatizer

_DENSE_INDEX_MODULE = None


def load_dense_index_module():
    """Load ``create_dense_index.py`` from this file's location, not the cwd."""
    global _DENSE_INDEX_MODULE
    if _DENSE_INDEX_MODULE is not None:
        return _DENSE_INDEX_MODULE

    module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_dense_index.py")
    spec = importlib.util.spec_from_file_location("wikifacts_create_dense_index", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load embedder profiles from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DENSE_INDEX_MODULE = module
    return module


def config_directory(config: str, rrf_k: Optional[int] = None) -> str:
    if config == "dense":
        return "dense"
    if config == "rrf":
        return f"rrf-k{rrf_k}"
    raise ValueError(f"Unknown retrieval config: {config}")


def sample_texts(texts: List[str], limit: int = 32) -> List[str]:
    return texts[:limit]


def sample_chunk_texts(chunks: Dict[str, List[str]], limit: int = 32) -> List[str]:
    sample: List[str] = []
    for fragments in chunks.values():
        for fragment in fragments:
            sample.append(fragment)
            if len(sample) >= limit:
                return sample
    return sample


def report_cache_sample(label: str, texts: List[str], cache: Optional[Dict[str, np.ndarray]], limit: int = 32) -> int:
    """Return how many sampled strings are missing from the cache."""
    sample = sample_texts(texts, limit)
    if not sample:
        print(f"{label}: no strings to check against the cache.")
        return 0
    if cache is None:
        print(
            f"{label}: cache is missing. All {len(sample)} sampled strings "
            "will be embedded with the selected profile."
        )
        return len(sample)
    hits = sum(text in cache for text in sample)
    print(f"{label}: {hits}/{len(sample)} sampled strings are keys in the cache.")
    if hits != len(sample):
        print(f"{label}: cache misses will be embedded with the selected profile.")
    return len(sample) - hits


def index_build_command(model_key: str, chunk_size: str) -> str:
    return (
        "python retrieval_pipeline/create_dense_index.py "
        f"--model {model_key} "
        f"--output-dir data/vector_store/{model_key}/{chunk_size}"
    )


# WIP: languages that have their own nltk tokenizer
LANG_MAPPING = {
    'ru': 'russian',
    'en': 'english',
}


class DenseRetriever:
    def __init__(self,
            profile,
            build_embedder,
            splitter: str,
            lang: str,
            device: str = 'cuda',
            backend: Optional[str] = None,
            corpus_cache: Optional[Dict[str, np.ndarray]] = None,
            query_cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        pooling = profile.pooling.lower()
        assert pooling in ("mean", "cls", "last"), "pooling must be mean, cls, or last"
        assert splitter in ("sentence", "paragraph", "article"), ""

        self.profile = profile
        self.build_embedder = build_embedder
        self.query_prefix = profile.query_prefix
        self.document_prefix = profile.document_prefix
        self.pooling = pooling
        self.backend = backend or profile.backend
        self.splitter = splitter
        self.lang = lang

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._embedder = None

        self.corpus_cache = corpus_cache
        self.query_cache = query_cache

        self.corpus_records = None
        self.corpus_embs = None
        self.corpus_embs_torch = None

    def _embed_misses(self, prefixed_texts: list[str]) -> np.ndarray:
        """Embed texts that already include the profile prefix."""
        if self._embedder is None:
            self._embedder = self.build_embedder(
                self.profile,
                self.backend,
                self.device,
                self.profile.batch_size,
                self.profile.max_len,
                self.profile.vllm_chunk_size,
            )
        return self._embedder.embed(prefixed_texts)

    def split_sentence(self, text: str) -> list[str]:
        return [sent.strip() for sent in sent_tokenize(text, language=LANG_MAPPING[self.lang])]

    def split_paragraph(self, text: str) -> list[str]:
        return [para.strip() for para in text.split("\n\n") if para.strip() != ""]

    def split(self, text: str) -> list[str]:
        if self.splitter == "sentence":
            return self.split_sentence(text)
        elif self.splitter == "paragraph":
            return self.split_paragraph(text)
        return [text]

    def get_embeddings(self, texts: list[str], prefix: str = "", cache: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Get embeddings for raw texts. Cache lookup uses the raw string.
        Misses are prefixed here, then passed to ``embedder.embed``.
        """
        results = [None] * len(texts)
        to_compute_indices = []
        to_compute_texts = []

        cache_hits = 0
        for i, text in enumerate(texts):
            if cache and text in cache:
                results[i] = cache[text]
                cache_hits += 1
            else:
                to_compute_indices.append(i)
                to_compute_texts.append(f"{prefix}{text}")

        if cache:
            hit_rate = (cache_hits / len(texts) * 100) if texts else 0
            print(f"Cache hits: {cache_hits}/{len(texts)} ({hit_rate:.2f}%)")

        if to_compute_texts:
            computed_embeddings = self._embed_misses(to_compute_texts)
            if len(computed_embeddings) != len(to_compute_indices):
                raise RuntimeError(
                    f"Embedder returned {len(computed_embeddings)} vectors for {len(to_compute_indices)} texts."
                )
            for idx, emb in zip(to_compute_indices, computed_embeddings):
                results[idx] = emb

        return np.vstack(results)

    def index_corpus(self, article_texts_by_id: Dict[str, List[str]]):
        """
        Pre-calculate and store embeddings for the entire corpus.
        This avoids re-assembling the corpus matrix for every query.
        """
        self.corpus_records = []
        for article_id, fragments in article_texts_by_id.items():
            for fragment in fragments:
                self.corpus_records.append({
                    'text': fragment,
                    'article_id': article_id,
                })
        
        if not self.corpus_records:
            print("Warning: Empty corpus passed to index_corpus.")
            return

        texts = [rec['text'] for rec in self.corpus_records]
        print(f"Assembling corpus matrix for {len(texts)} fragments...")
        self.corpus_embs = self.get_embeddings(texts, prefix=self.document_prefix, cache=self.corpus_cache)
        
        # Move to device if possible to speed up cosine similarity
        if self.device != 'cpu':
            try:
                self.corpus_embs_torch = torch.from_numpy(self.corpus_embs).to(self.device)
                print(f"Moved corpus embeddings matrix to {self.device}")
            except Exception as e:
                print(f"Warning: Could not move corpus embeddings to GPU: {e}")
                self.corpus_embs_torch = None
        else:
            self.corpus_embs_torch = None

    def retrieve(self, fact: str, article_texts_by_id: Dict[str, Any] = None, top_k: int = 5, use_presplit_chunks: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top fragments across multiple articles using dense embeddings.

        Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
        """
        # 1. Use pre-indexed corpus if available
        if self.corpus_embs is not None and self.corpus_records is not None:
            fragment_records = self.corpus_records
            
            query_emb = self.get_embeddings([fact], prefix=self.query_prefix, cache=self.query_cache)
            
            if self.corpus_embs_torch is not None:
                # Optimized GPU similarity (matrix multiplication)
                query_emb_torch = torch.from_numpy(query_emb).to(self.device)
                # Compute dot products (equivalent to cosine similarity for normalized vectors)
                sims = torch.mm(query_emb_torch, self.corpus_embs_torch.T)[0].cpu().numpy()
            else:
                sims = cosine_similarity(query_emb, self.corpus_embs)[0]
        
        # 2. Fallback to dynamic assembly (original logic)
        else:
            if article_texts_by_id is None:
                raise ValueError("article_texts_by_id must be provided if corpus is not indexed.")
            
            fragment_records = []
            for article_id, article_text in article_texts_by_id.items():
                if not use_presplit_chunks:
                    fragments = self.split(article_text)
                else:
                    fragments = article_text

                for fragment in fragments:
                    fragment_records.append({
                        'text': fragment,
                        'article_id': article_id,
                    })

            if not fragment_records:
                return []

            query_emb = self.get_embeddings([fact], prefix=self.query_prefix, cache=self.query_cache)
            frag_texts = [rec['text'] for rec in fragment_records]
            frag_embs = self.get_embeddings(frag_texts, prefix=self.document_prefix, cache=self.corpus_cache)

            sims = cosine_similarity(query_emb, frag_embs)[0]

        # 3. Process results (common logic)
        order = sims.argsort()[::-1]
        top_idx = order[:min(top_k, len(order))]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            rec = fragment_records[idx]
            results.append({
                'text': rec['text'],
                'score': float(sims[idx]),
                'article_id': rec['article_id'],
            })

        return results


class BM25Retriever:
    """
    BM25 retriever.

    The input corpus can be provided:
    1. at the initialization stage - when the corpus is static across queries.
    2. at the retrieval stage - when the corpus is dynamic and changes per query.

    Corpus, provided at the retrieval stage, will have a priority over the one at the initialization stage.
    """

    def __init__(self,
            lang: str,
            splitter: str,
            save_dir: str = os.path.join(REPO_ROOT, 'data', 'vector_store', 'bm25', 'articles'),
            corpus: Optional[Dict[str, Any]] = None,
            reindex_corpus: bool = False,
            k1: float = 1.2,
            b: float = 0.75,
    ):
        assert splitter in ("sentence", "paragraph", "article"), ""

        self.splitter = splitter
        self.lang = lang
        self.tokenizer = MultilingualLemmatizer(lang)
        self.k1 = k1
        self.b = b
        self.save_dir = save_dir

        self._index = []
        self._texts = []
        self._owners = []
        self.model = None

        if corpus:
            if reindex_corpus:
                self.create_index(corpus)
            else:
                self.load_index()

    def create_index(self, corpus: Optional[Dict[str, Any]] = None):
        for article_id, article_text in tqdm(corpus.items(), desc="Creating BM25 index"):
            if isinstance(article_text, list):
                frags = article_text
            else:
                frags = self.split(article_text)

            self._owners.extend([article_id] * len(frags))
            self._texts.extend(frags)
            self._index.extend([self._tokenize(f) for f in frags])

        self.model = BM25(k1=self.k1, b=self.b)
        self.model.index(self._index)

        self.save_index()

    def save_index(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model.save(save_dir=self.save_dir)

        with open(os.path.join(self.save_dir, 'index.pkl'), 'wb') as f:
            pickle.dump(self._index, f)
        with open(os.path.join(self.save_dir, 'texts.pkl'), 'wb') as f:
            pickle.dump(self._texts, f)
        with open(os.path.join(self.save_dir, 'owners.pkl'), 'wb') as f:
            pickle.dump(self._owners, f)

        print(f'Successfully saved BM25 index at path: {self.save_dir}')

    def load_index(self):
        self.model = BM25.load(save_dir=self.save_dir)

        with open(os.path.join(self.save_dir, 'index.pkl'), 'rb') as f:
            self._index = pickle.load(f)
        with open(os.path.join(self.save_dir, 'texts.pkl'), 'rb') as f:
            self._texts = pickle.load(f)
        with open(os.path.join(self.save_dir, 'owners.pkl'), 'rb') as f:
            self._owners = pickle.load(f)

        print(f'Successfully loaded BM25 index at path: {self.save_dir}')

    def split_sentence(self, text: str) -> list[str]:
        return [sent.strip() for sent in sent_tokenize(text, language=LANG_MAPPING[self.lang])]

    def split_paragraph(self, text: str) -> list[str]:
        return [para.strip() for para in text.split("\n\n") if para.strip() != ""]

    def split(self, text: str) -> list[str]:
        if self.splitter == "sentence":
            return self.split_sentence(text)
        elif self.splitter == "paragraph":
            return self.split_paragraph(text)
        return [text]

    def _tokenize(self, text: str) -> List[str]:
        # Use lemmatizer to normalize; then whitespace split to tokens
        normalized = self.tokenizer.lemmatize_text(text, remove_stopwords=True)
        return [tok for tok in normalized.split() if tok]

    def retrieve(self, fact: str, article_texts_by_id: Dict[str, Any] = {}, top_k: int = 5, use_presplit_chunks: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve top fragments across multiple articles using BM25.

        Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
        """
        # Dynamic corpus
        if article_texts_by_id:
            fragments = []
            owners = []
            for article_id, article_text in article_texts_by_id.items():
                if not use_presplit_chunks:
                    frags = self.split(article_text)
                else:
                    frags = article_text
                fragments.extend(frags)
                owners.extend([article_id] * len(frags))

            if not fragments:
                return []

            k = min(top_k, len(fragments))

            tokenized_docs = [self._tokenize(f) for f in fragments]
            bm25 = BM25(k1=self.k1, b=self.b)
            bm25.index(tokenized_docs)

            tokenized_query = self._tokenize(fact)
            indices, scores = bm25.retrieve([tokenized_query], k=k)
            top_idx = indices[0]
            top_scores = scores[0]

            results: List[Dict[str, Any]] = []
            for i, idx in enumerate(top_idx):
                results.append({
                    'text': fragments[idx],
                    'score': float(top_scores[i]),
                    'article_id': owners[idx],
                })

            return results
        # Static corpus
        else:
            if not self.model:
                raise ValueError("BM25 model was not initialized for static retrieval.")

            k = min(top_k, len(self._index))

            tokenized_query = self._tokenize(fact)
            indices, scores = self.model.retrieve([tokenized_query], k=k)
            top_idx = indices[0] 
            top_scores = scores[0]

            results: List[Dict[str, Any]] = []
            for i, idx in enumerate(top_idx):
                results.append({
                    'text': self._texts[idx],
                    'score': float(top_scores[i]),
                    'article_id': self._owners[idx],
                })

            return results
    

class BM25AndDenseRRFRetriever:
    """
    Reciprocal Rank Fusion (RRF) retriever combining BM25 and Dense retrieval.
    """
    def __init__(self, 
                 bm25_retriever: BM25Retriever, 
                 dense_retriever: DenseRetriever,
                 chunks: Dict[str, List[str]],
                 k: int = 60):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.chunks = chunks
        self.k = k

        # Pre-index dense corpus once during initialization
        self.dense_retriever.index_corpus(self.chunks)

    def retrieve(self, fact: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """
        Independent retrievals via BM25 and Dense, merged via RRF.
        """
        # Retrieve top 200 from each
        bm25_res = self.bm25_retriever.retrieve(fact, top_k=200)
        dense_res = self.dense_retriever.retrieve(
            fact, 
            top_k=200, 
            use_presplit_chunks=True
        )

        doc_scores = {}
        # Key is (article_id, text) to uniquely identify a fragment
        for rank, res in enumerate(bm25_res, 1):
            doc_id = (res['article_id'], res['text'])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)
            
        for rank, res in enumerate(dense_res, 1):
            doc_id = (res['article_id'], res['text'])
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (self.k + rank)

        # Reconstruct result list with merged scores
        merged_results = []
        for (article_id, text), rrf_score in doc_scores.items():
            merged_results.append({
                'article_id': article_id,
                'text': text,
                'rrf_score': rrf_score
            })

        # Sort by RRF score descending
        merged_results.sort(key=lambda x: x['rrf_score'], reverse=True)

        return merged_results[:top_k]


class DenseOnlyRetriever:
    """Dense search whose cosine similarity is stored in ``rrf_score``."""

    def __init__(self, dense_retriever: DenseRetriever, chunks: Dict[str, List[str]]):
        self.dense_retriever = dense_retriever
        self.dense_retriever.index_corpus(chunks)

    def retrieve(self, fact: str, top_k: int = 100) -> List[Dict[str, Any]]:
        hits = self.dense_retriever.retrieve(fact, top_k=top_k, use_presplit_chunks=True)
        return [
            {
                'text': item['text'],
                'rrf_score': float(item['score']),
                'article_id': item['article_id'],
            }
            for item in hits
        ]


def load_local_data(dataset_dir: str, subset: str) -> Dict[str, str]:
    """
    Load a subset (corpus or queries) from a local dataset directory.
    Assumes files are named {subset}.jsonl inside the directory.
    """
    path = os.path.join(dataset_dir, f"{subset}.jsonl")
    ds = load_dataset("json", data_files=path, split="train")
    return {record['_id']: record['text'] for record in ds}


def load_hf_data(dataset_name: str, split: str='queries') -> Dict[str, str]:
    ds = load_dataset(dataset_name, split)
    ds_dict = {}
    for record in ds['train']:
        ds_dict[record['_id']] = record['text']
    return ds_dict


def load_cache(path: str) -> Optional[Dict[str, np.ndarray]]:
    emb_path = os.path.join(path, 'embeddings.npy')
    meta_path = os.path.join(path, 'metadata.json')
    if not os.path.exists(emb_path) or not os.path.exists(meta_path):
        print(f"Cache not found at {path}")
        return None

    print(f"Loading cache from {path}...")
    embeddings = np.load(emb_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        texts = json.load(f)

    return {text: emb for text, emb in zip(texts, embeddings)}
    

def retrieve_from_corpus(
    queries_dataset,
    retriever,
    top_k: int = 100,
):
    retrieval_results = {}

    for qid, qtext in tqdm(queries_dataset.items(), desc="Retrieving relevant fragments from corpus"):
        top = retriever.retrieve(
            qtext, 
            top_k=top_k, 
        )

        items = []
        for item in top:
            items.append({
                'text': item['text'],
                'rrf_score': float(item['rrf_score']),
                'article_id': item['article_id'],
            })

        retrieval_results[qid] = items

    return retrieval_results


def main():
    import argparse

    class _MarkRrfK(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.rrf_k = values
            namespace.rrf_k_set = True

    indexer = load_dense_index_module()
    model_keys = list(indexer.PROFILES)

    parser = argparse.ArgumentParser(
        description="Dense retrieval, or BM25 and dense retrieval fused with RRF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=model_keys, default="e5", help="Embedder profile key.")
    parser.add_argument("--config", choices=["dense", "rrf"], default="rrf", help="dense searches with cosine similarity. rrf fuses BM25 and dense.")
    parser.add_argument(
        "--rrf-k",
        action=_MarkRrfK,
        type=int,
        choices=[0, 60],
        default=60,
        help="RRF constant when --config rrf. Rejected when --config dense.",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Backend used only when a cache miss has to be embedded. Defaults to the profile backend.",
    )
    parser.add_argument("--chunk_size", type=str, default="sentences", help="Chunk size (e.g., sw_4)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for a cache-miss embedder")
    parser.add_argument("--local_dataset_path", type=str, default="path/to/local/dataset", help="Path to local dataset directory")
    parser.add_argument("--reindex_bm25", action="store_true", help="Reindex BM25")
    args = parser.parse_args()

    if args.config == "dense" and getattr(args, "rrf_k_set", False):
        parser.error("--rrf-k cannot be used with --config dense")

    global STAGE_2_SIZE
    STAGE_2_SIZE = args.chunk_size
    profile = indexer.PROFILES[args.model]
    config_name = config_directory(args.config, args.rrf_k)

    # 1. Load data
    print("Loading datasets...")
    # Using local data as requested. Placeholders provided in default args.
    # corpus_dataset = load_local_data(args.local_corpus_path)
    # queries_dataset = load_local_data(args.local_queries_path)
    
    # Fallback to HF if local path is default/placeholder for now, 
    # but the logic is set up for local.
    if args.local_dataset_path == "path/to/local/dataset":
        print("Using placeholder path, loading from HF as fallback...")
        corpus_dataset = load_hf_data("kaengreg/wikifacts-articles", split="corpus")
        queries_dataset = load_hf_data("kaengreg/wikifacts-articles", split="queries")
    else:
        corpus_dataset = load_local_data(args.local_dataset_path, "corpus")
        queries_dataset = load_local_data(args.local_dataset_path, "queries")

    # 2. Load heavy artifacts
    print(f"Loading heavy artifacts for STAGE_2_SIZE={STAGE_2_SIZE}...")
    with open(os.path.join(REPO_ROOT, 'heavy_artifacts', 'retrieval', STAGE_2_SIZE, 'ready_chunks.json'), 'r') as f:
        chunks = json.load(f)

    vector_store = os.path.join(REPO_ROOT, 'data', 'vector_store', profile.key, STAGE_2_SIZE)
    corpus_cache = load_cache(f'{vector_store}/corpus')
    query_cache = load_cache(f'{vector_store}/queries')
    chunk_sample = sample_chunk_texts(chunks)
    query_sample = sample_texts(list(queries_dataset.values()))
    missed = report_cache_sample("chunks", chunk_sample, corpus_cache)
    missed += report_cache_sample("queries", query_sample, query_cache)
    if missed:
        print("Build a matching index from the repository root:")
        print(index_build_command(profile.key, STAGE_2_SIZE))

    # 3. Initialize retrievers
    print("Initializing retrievers...")
    dense_retriever = DenseRetriever(
        profile=profile,
        build_embedder=indexer.build_embedder,
        lang='ru',
        splitter='sentence',
        device=args.device,
        backend=args.backend,
        corpus_cache=corpus_cache,
        query_cache=query_cache,
    )

    if args.config == "dense":
        retriever = DenseOnlyRetriever(dense_retriever, chunks)
    else:
        bm25_retriever = BM25Retriever(
            lang='ru',
            splitter='sentence',
            save_dir=os.path.join(REPO_ROOT, 'data', 'vector_store', 'bm25', STAGE_2_SIZE),
            corpus=chunks, # Initialize on sentence level chunks
            reindex_corpus=args.reindex_bm25,
        )
        retriever = BM25AndDenseRRFRetriever(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            chunks=chunks,
            k=args.rrf_k,
        )

    # 4. Run inference
    print("\n--- Starting inference ---")
    results = retrieve_from_corpus(
        queries_dataset=queries_dataset,
        retriever=retriever,
        top_k=100
    )

    # 5. Save results
    out_dir = os.path.join(REPO_ROOT, 'heavy_artifacts', 'retrieval', STAGE_2_SIZE, profile.key, config_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'results.jsonl')

    with open(out_path, 'w') as f:
        for qid, items in results.items():
            rec = {'query_id': qid, 'results': items}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
