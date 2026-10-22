"""Two-stage retrieval: BM25 over articles, then dense search over their chunks.

Run this file from ``notebooks/``. The cascade, the penalty grid, and the
RRF weights stay as they are. ``--model`` selects the dense embedder.

Dense vectors are read from
``../data/vector_store/<model>/<chunk_size>/{corpus,queries}``.
Build that layout from the repository root:

    python utils/dataset-enhancement/create_dense_index.py \\
      --model <profile key> \\
      --output-dir data/vector_store/<profile key>/<chunk_size>

A cache hit returns the stored vector. A miss is embedded with that
profile's prefixes. Results are written under
``../heavy_artifacts/retrieval/<chunk_size>/<model>/<penalty config>/``.
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

from datasets import load_dataset

import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
from nltk import sent_tokenize

from bm25s import BM25

from lemmatizer import MultilingualLemmatizer

_DENSE_INDEX_MODULE = None


def load_dense_index_module():
    """Load ``create_dense_index.py`` from this file's location, not the cwd."""
    global _DENSE_INDEX_MODULE
    if _DENSE_INDEX_MODULE is not None:
        return _DENSE_INDEX_MODULE

    module_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "utils",
            "dataset-enhancement",
            "create_dense_index.py",
        )
    )
    spec = importlib.util.spec_from_file_location("wikifacts_create_dense_index", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load embedder profiles from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DENSE_INDEX_MODULE = module
    return module


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
    sample = texts[:limit]
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
        "python utils/dataset-enhancement/create_dense_index.py "
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
        self.device = device or "cpu"
        self._embedder = None

        self.corpus_cache = corpus_cache
        self.query_cache = query_cache

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

        for i, text in enumerate(texts):
            if cache and text in cache:
                results[i] = cache[text]
            else:
                to_compute_indices.append(i)
                to_compute_texts.append(f"{prefix}{text}")

        if to_compute_texts:
            computed_embeddings = self._embed_misses(to_compute_texts)
            if len(computed_embeddings) != len(to_compute_indices):
                raise RuntimeError(
                    f"Embedder returned {len(computed_embeddings)} vectors for {len(to_compute_indices)} texts."
                )
            for idx, emb in zip(to_compute_indices, computed_embeddings):
                results[idx] = emb

        return np.vstack(results)

    def retrieve(self, fact: str, article_texts_by_id: Dict[str, str], top_k: int = 5, use_presplit_chunks: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve top fragments across multiple articles using dense embeddings.

        Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
        """
        fragment_records: List[Dict[str, Any]] = []
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
            corpus: Optional[Dict[str, str]] = None,
            reindex_corpus: bool = True,
            k1: float = 1.2,
            b: float = 0.75,
    ):
        assert splitter in ("sentence", "paragraph", "article"), ""

        self.splitter = splitter
        self.lang = lang
        self.tokenizer = MultilingualLemmatizer(lang)
        self.k1 = k1
        self.b = b

        self._index = []
        self._texts = []
        self._owners = []
        self.model = None

        if corpus:
            if reindex_corpus:
                self.create_index(corpus)
            else:
                self.load_index()

    def create_index(self, corpus: Optional[Dict[str, str]] = None):
        for article_id, article_text in tqdm(corpus.items(), desc="Creating BM25 index"):
            frags = self.split(article_text)

            self._owners.extend([article_id] * len(frags))
            self._texts.extend(frags)
            self._index.extend([self._tokenize(f) for f in frags])

        self.model = BM25(k1=self.k1, b=self.b)
        self.model.index(self._index)

        self.save_index()

    def save_index(self):
        save_dir = '../data/vector_store/bm25/articles'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.model.save(save_dir=save_dir)

        with open(os.path.join(save_dir, 'index.pkl'), 'wb') as f:
            pickle.dump(self._index, f)
        with open(os.path.join(save_dir, 'texts.pkl'), 'wb') as f:
            pickle.dump(self._texts, f)
        with open(os.path.join(save_dir, 'owners.pkl'), 'wb') as f:
            pickle.dump(self._owners, f)

        print(f'Successfully saved BM25 index at path: {save_dir}')

    def load_index(self):
        save_dir = '../data/vector_store/bm25/articles'

        self.model = BM25.load(save_dir=save_dir)

        with open(os.path.join(save_dir, 'index.pkl'), 'rb') as f:
            self._index = pickle.load(f)
        with open(os.path.join(save_dir, 'texts.pkl'), 'rb') as f:
            self._texts = pickle.load(f)
        with open(os.path.join(save_dir, 'owners.pkl'), 'rb') as f:
            self._owners = pickle.load(f)

        print(f'Successfully loaded BM25 index at path: {save_dir}')

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

    def retrieve(self, fact: str, article_texts_by_id: Dict[str, str] = {}, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top fragments across multiple articles using BM25.

        Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
        """
        # Dynamic corpus
        if article_texts_by_id:
            fragments = []
            owners = []
            for article_id, article_text in article_texts_by_id.items():
                frags = self.split(article_text)
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
    

class PenaltyScorer:
    """
    Encapsulates all penalty calculations for articles and sentences.
    """
    def __init__(self, avg_article_len: int = 15060):
        self.avg_article_len = avg_article_len

    def get_articles_len_penalties_global(self, articles: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate length penalties for retrieved articles shorter than global average.
        """
        penalties = {}
        for idx, text in articles.items():
            article_len = len(text.strip())
            if article_len >= self.avg_article_len:
                penalties[idx] = 1.0
            else:
                penalties[idx] = 1.0 - ((self.avg_article_len - article_len) / self.avg_article_len)
        return penalties

    def get_articles_len_penalties_local(self, articles_sentences: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate length penalties for long retrieved articles (based on sentence count).
        """
        if not articles_sentences:
            return {}
        
        min_len, sum_len = -1, 0
        for sentences in articles_sentences.values():
            cur_len = len(sentences)
            if cur_len < min_len or min_len == -1:
                min_len = cur_len
            sum_len += cur_len

        penalties = {}
        for idx, sentences in articles_sentences.items():
            if sum_len == 0:
                penalties[idx] = 1.0
            else:
                penalties[idx] = max(0.7, 1.0 - ((len(sentences) - min_len) / sum_len))
        return penalties

    def get_sentence_len_rel_penalty(self, sentence: str, fact: str) -> float:
        """
        Calculate length penalty for the sentence relative to the fact's length.
        """
        sent_len = len(sentence)
        fact_len = len(fact)
        len_dist = fact_len - sent_len
        if len_dist > 0:
            return 1.0 - (len_dist / fact_len)
        return 1.0

    def get_sentence_len_abs_penalty(self, sentence: str) -> float:
        """
        Calculate absolute length penalty for the sentence.
        """
        sent_len = len(sentence)
        if 80 <= sent_len <= 140:
            return 1.0
        if 60 <= sent_len <= 155:
            return 0.9
        if 50 <= sent_len <= 190:
            return 0.8
        if 35 <= sent_len <= 230:
            return 0.75
        return 0.7


class TwoStageRetrieverRRF:
    """
    Two-stage retriever with Reciprocal Rank Fusion between stages.
    """

    def __init__(self, 
                 bm25_retriever: BM25Retriever, 
                 dense_retriever: DenseRetriever,
                 penalty_scorer: PenaltyScorer,
                 chunks: Dict[str, list[str]],
                 penalty_config: str = 'no_penalties'):
        """
        :param bm25_retriever: Pre-initialized BM25Retriever
        :param dense_retriever: Pre-initialized DenseRetriever
        :param penalty_scorer: Pre-initialized PenaltyScorer
        :param chunks: Pre-split chunks for the second retrieval stage.
        :param penalty_config: Config for penalties ('no_penalties', 'article_penalties', 'sentence_penalties', 'both_penalties')
        """
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.penalty_scorer = penalty_scorer
        self.chunks = chunks
        self.penalty_config = penalty_config

    def _rerank_with_rrf(self,
                         docs: List[Dict[str, Any]],
                         weights: List[float] = [0.5, 0.5],
                         len_penalties: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Rerank retrieved documents using Weighted Reciprocal Rank Fusion.
        """
        if not len_penalties:
            len_penalties = {entry['article_id']: 1.0 for entry in docs}

        for entry in docs:
            article_penalty = len_penalties.get(entry['article_id'], 1.0)
            rrf_score = (
                weights[0] * article_penalty * entry['article_reciprocal_rank'] 
                + weights[1] * entry['sentence_penalty'] * entry['sentence_reciprocal_rank']
            )
            entry['rrf_score'] = rrf_score

        return sorted(docs, key=lambda x: x['rrf_score'], reverse=True)

    def retrieve(self, 
                 fact: str, 
                 top_k_articles: int = 3, 
                 top_k_sentences: int = 10,
                 stages_weights: List[float] = [0.5, 0.5]) -> List[Dict[str, Any]]:
        """
        Two-stage retrieval process.
        """
        # Stage 1: Retrieve top-k articles using BM25
        stage1_results = self.bm25_retriever.retrieve(
            fact, 
            article_texts_by_id={}, 
            top_k=top_k_articles
        )

        if not stage1_results:
            return []

        # Calculate reciprocal ranks for 1st stage
        article_to_rank = {}
        for idx, entry in enumerate(stage1_results, 1):
            entry_id = entry['article_id']
            if entry_id not in article_to_rank:
                article_to_rank[entry_id] = 1 / idx

        relevant_article_ids = list(article_to_rank.keys())

        # Build subcorpus with only the chunks from relevant articles
        subcorpus = {}
        article_subcorpus_full = {}
        article_subcorpus_sents = {}
        total_sentences = 0
        for article_id in relevant_article_ids:
            if article_id in self.chunks:
                article_sentences = list(set(self.chunks[article_id]))
                subcorpus[article_id] = article_sentences
                article_subcorpus_full[article_id] = ' '.join(article_sentences)
                article_subcorpus_sents[article_id] = article_sentences
                total_sentences += len(article_sentences)

        # Calculate article length penalties if requested
        len_penalties = None
        if self.penalty_config in ('article_penalties', 'both_penalties'):
            len_penalties = self.penalty_scorer.get_articles_len_penalties_local(article_subcorpus_sents)
            # len_penalties = self.penalty_scorer.get_articles_len_penalties_global(article_subcorpus_full)

        # Stage 2: Use DenseRetriever on the subcorpus to retrieve sentences
        stage2_results = self.dense_retriever.retrieve(
            fact, 
            article_texts_by_id=subcorpus, 
            top_k=total_sentences, 
            use_presplit_chunks=True
        )

        # Enrich the results with reciprocal ranks for both stages + penalties
        for idx, entry in enumerate(stage2_results, 1):
            entry['article_reciprocal_rank'] = article_to_rank[entry['article_id']]
            entry['sentence_reciprocal_rank'] = 1 / idx
            
            if self.penalty_config in ('sentence_penalties', 'both_penalties'):
                entry['sentence_penalty'] = self.penalty_scorer.get_sentence_len_rel_penalty(entry['text'], fact)
            else:
                entry['sentence_penalty'] = 1.0

        # Rerank results using weighted RRF
        rrf_reranked_results = self._rerank_with_rrf(
            stage2_results, 
            weights=stages_weights,
            len_penalties=len_penalties,
        )

        return rrf_reranked_results[:top_k_sentences]


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
    retriever: TwoStageRetrieverRRF,
    top_k_articles: int,
    stages_weights: List[float],
):
    retrieval_results = {}

    for qid, qtext in tqdm(queries_dataset.items(), desc="Retrieving relevant fragments from corpus"):
        top = retriever.retrieve(
            qtext, 
            top_k_articles=top_k_articles, 
            top_k_sentences=100, 
            stages_weights=stages_weights,
        )

        items = []
        for item in top:
            items.append({
                'text': item['text'],
                'rrf_score': float(item['rrf_score']),
                'article_id': item['article_id'],
                'article_reciprocal_rank': float(item['article_reciprocal_rank']),
                'sentence_reciprocal_rank': float(item['sentence_reciprocal_rank']),
                'sentence_penalty': float(item['sentence_penalty']),
            })

        retrieval_results[qid] = items

    return retrieval_results


def main():
    import argparse

    indexer = load_dense_index_module()
    model_keys = list(indexer.PROFILES)

    parser = argparse.ArgumentParser(
        description="Two-stage retrieval with RRF and penalties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=model_keys, default="e5", help="Embedder profile key.")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Backend used only when a cache miss has to be embedded. Defaults to the profile backend.",
    )
    parser.add_argument("--chunk_size", type=str, default="sentences", help="Chunk size for 2nd stage (e.g., sw_4)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for a cache-miss embedder")
    parser.add_argument("--corpus_name", type=str, default="kaengreg/wikifacts-articles", help="HF dataset name")
    args = parser.parse_args()

    global STAGE_2_SIZE
    STAGE_2_SIZE = args.chunk_size
    profile = indexer.PROFILES[args.model]

    # 1. Load data
    print("Loading datasets...")
    corpus_dataset = load_hf_data(args.corpus_name, split="corpus")
    queries_dataset = load_hf_data(args.corpus_name, split="queries")

    # 2. Pre-calculate values
    avg_article_len = int(np.mean([len(x.strip()) for x in corpus_dataset.values()]))
    penalty_scorer = PenaltyScorer(avg_article_len)

    # 3. Load heavy artifacts
    print(f"Loading heavy artifacts for STAGE_2_SIZE={STAGE_2_SIZE}...")
    with open(f'../heavy_artifacts/retrieval/{STAGE_2_SIZE}/ready_chunks.json', 'r') as f:
        chunks = json.load(f)

    vector_store = f'../data/vector_store/{profile.key}/{STAGE_2_SIZE}'
    corpus_cache = load_cache(f'{vector_store}/corpus')
    query_cache = load_cache(f'{vector_store}/queries')
    missed = report_cache_sample("chunks", sample_chunk_texts(chunks), corpus_cache)
    missed += report_cache_sample("queries", list(queries_dataset.values()), query_cache)
    if missed:
        print("Build a matching index from the repository root:")
        print(index_build_command(profile.key, STAGE_2_SIZE))

    # 4. Initialize retrievers (once)
    print("Initializing retrievers...")
    bm25_retriever = BM25Retriever(
        lang='ru',
        splitter='article',
        corpus=corpus_dataset,
        reindex_corpus=False,
    )

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

    # 5. Define grids
    top_k_articles_grid = [1, 2, 3, 4, 5, 7, 10]
    # top_k_articles_grid = [1]
    weights_grid = [[0.0, 1.0], [0.2, 0.8], [0.25, 0.75], [0.3, 0.7], [0.35, 0.65], [0.4, 0.6]]
    # weights_grid = [[0.0, 1.0]]
    # penalty_configs = ['no_penalties', 'article_penalties', 'sentence_penalties', 'both_penalties']
    penalty_configs = ['both_penalties']

    # 6. Run inference for all configurations
    for p_config in penalty_configs:
        print(f"\n--- Starting inference for penalty config: {p_config} ---")
        
        # Initialize two-stage retriever with current penalty config
        retriever = TwoStageRetrieverRRF(
            bm25_retriever=bm25_retriever,
            dense_retriever=dense_retriever,
            penalty_scorer=penalty_scorer,
            chunks=chunks,
            penalty_config=p_config
        )

        for t_i in top_k_articles_grid:
            for w_i in weights_grid:
                print(f'Config: {p_config}, top_k_articles = {t_i}, weights = {w_i}')
                
                results = retrieve_from_corpus(
                    queries_dataset=queries_dataset,
                    retriever=retriever,
                    top_k_articles=t_i,
                    stages_weights=w_i
                )

                # Automated out path
                out_dir = f'../heavy_artifacts/retrieval/{STAGE_2_SIZE}/{profile.key}/{p_config}'
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'{t_i}_{w_i[0]}_{w_i[1]}.jsonl')

                with open(out_path, 'w') as f:
                    for qid, items in results.items():
                        rec = {'query_id': qid, 'results': items}
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
                print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()
