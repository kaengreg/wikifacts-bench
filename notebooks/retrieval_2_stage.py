import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    from typing import List, Dict, Any, Optional

    from tqdm import tqdm

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity 

    import torch 
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    import nltk
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    from nltk import sent_tokenize

    from bm25s import BM25

    from lemmatizer import MultilingualLemmatizer


    # WIP: languages that have their own nltk tokenizer
    LANG_MAPPING = {
        'ru': 'russian',
        'en': 'english',
    }


    class DenseRetriever:
        def __init__(self,
                model_name: str,
                maxlen: int, 
                pooling: str,
                splitter: str,
                lang: str,
                device: str = 'cuda',
                batch_size: int = 16,
        ): 
            assert pooling in ("mean", "cls"), "pooling must be either mean or cls"
            assert splitter in ("sentence", "paragraph", "article"), ""

            self.splitter = splitter
            self.lang = lang
        
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, self.tokenizer = self.load_model(model_name, self.device)

            self.maxlen = maxlen
            self.batch_size = batch_size
            self.pooling = pooling.lower()
        

        def load_model(self, model_name: str, device: str = 'cuda'):
            model = AutoModel.from_pretrained(model_name).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

            return model, tokenizer

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

        def _average_pool(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            last_hidden_states = model_output.last_hidden_state
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        def _cls_pool(self, model_output: torch.Tensor) -> torch.Tensor:
            return model_output.last_hidden_state[:, 0, :]

        def get_embeddings(self, texts: list[str]) -> np.ndarray:
            embeddings = []
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing Batches"):
                batch_texts = texts[i:i + self.batch_size]
                batch_dict = self.tokenizer(batch_texts, max_length=self.maxlen, padding=True, truncation=True,
                                            return_tensors='pt')
                batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                if self.pooling == 'mean':
                    batch_embeddings = self._average_pool(outputs, batch_dict['attention_mask'])
                elif self.pooling == 'cls':
                    batch_embeddings = self._cls_pool(outputs)
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")

                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

            return np.vstack(embeddings)

        def retrieve(self, fact: str, article_texts_by_id: Dict[str, str], top_k: int = 5) -> List[Dict[str, Any]]:
            """
            Retrieve top fragments across multiple articles using dense embeddings.

            Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
            """
            fragment_records: List[Dict[str, Any]] = []
            for article_id, article_text in article_texts_by_id.items():
                fragments = self.split(article_text)
                for fragment in fragments:
                    fragment_records.append({
                        'text': fragment,
                        'article_id': article_id,
                    })

            if not fragment_records:
                return []

            query_emb = self.get_embeddings([fact])
            frag_texts = [rec['text'] for rec in fragment_records]
            frag_embs = self.get_embeddings(frag_texts)

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
                self.create_index(corpus)

        def create_index(self, corpus: Optional[Dict[str, str]] = None):
            for article_id, article_text in corpus.items():
                frags = self.split(article_text)

                self._owners.extend([article_id] * len(frags))
                self._texts.extend(frags)
                self._index.extend([self._tokenize(f) for f in frags])

            self.model = BM25(k1=self.k1, b=self.b)
            self.model.index(self._index)

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


    class RelevantRetriever:
        """
        ### Description

        Aggregating, high-level retriever with a unified API.

        This class wraps a sparse BM25 retriever and a dense embedding retriever 
        and exposes a single ``retrieve`` method. 
    
        By default, it uses the sparse BM25 retriever. 
        Set ``mode='dense'`` to use the dense retriever.

        ### Parameters

        :param mode: Which retriever to initialize (``'sparse'`` or ``'dense'``), defaults to ``'sparse'``.
        :type mode: str, optional
        :param splitter: Granularity for splitting article text (``'sentence'``, ``'paragraph'``, or ``'article'``), defaults to ``'sentence'``.
        :type splitter: str, optional
        :param extra_kwargs: Parameters override for retriever initializer. Keys vary depending on the selected mode.
        :type extra_kwargs: dict, optional

        ### Extra arguments

        Mode-specific options passed via ``extra_kwargs``:

        - When ``mode='sparse'`` (BM25):

          - ``lang`` (str): Language code used by the lemmatizer/tokenizer, defaults to ``'ru'``.
          - ``k1`` (float): BM25 k1 parameter, defaults to ``1.2``.
          - ``b`` (float): BM25 b parameter, defaults to ``0.75``.

        - When ``mode='dense'``:

          - ``model_name`` (str): HF encoder model to embed text, defaults to ``'intfloat/multilingual-e5-large'``.
          - ``maxlen`` (int): Max sequence length for tokenizer truncation, defaults to ``256``.
          - ``pooling`` (str): Pooling strategy (``'mean'`` or ``'cls'``), defaults to ``'mean'``.
          - ``device`` (str or None): ``'cuda'`` or ``'cpu'``; ``None`` auto-selects, defaults to ``None``.
          - ``batch_size`` (int): Batch size for embedding computation, defaults to ``16``.

        ### Examples

        The following examples demonstrate how to use the ``RelevantRetriever`` class::

            # Default sparse BM25
            retriever = RelevantRetriever(mode='sparse', splitter='sentence')
            top_fragments = retriever.retrieve(fact, article_texts_by_id, top_k=5)

            # Sparse BM25 with language override
            retriever = RelevantRetriever(mode='sparse', splitter='sentence', extra_kwargs={'lang': 'en'})
            top_fragments = retriever.retrieve(fact, article_texts_by_id, top_k=8)

            # Dense with defaults
            retriever = RelevantRetriever(mode='dense', splitter='sentence')
            top_fragments = retriever.retrieve(fact, article_texts_by_id, top_k=5)

            # Dense with custom model/device
            retriever = RelevantRetriever(
                mode='dense', splitter='sentence',
                extra_kwargs={'model_name': 'intfloat/multilingual-e5-large', 'device': 'cuda', 'maxlen': 256},
            )
            top_fragments = retriever.retrieve(fact, article_texts_by_id, top_k=10)
        """
        def __init__(self, mode: str = 'sparse', splitter: str = 'sentence', extra_kwargs: dict = {}):
            self.mode = mode.lower()
            self.splitter = splitter.lower()

            # Defaults per mode
            if self.mode == 'sparse':
                defaults = {
                    'lang': 'ru',
                    'k1': 1.2,
                    'b': 0.75,
                }
                init_kwargs = {**defaults, **extra_kwargs}
                init_kwargs['splitter'] = splitter
                self._implementation = BM25Retriever(**init_kwargs)
            elif self.mode == 'dense':
                defaults = {
                    'model_name': 'intfloat/multilingual-e5-large',
                    'maxlen': 256,
                    'pooling': 'mean',
                    'device': None,
                    'batch_size': 16,
                }
                init_kwargs = {**defaults, **extra_kwargs}
                init_kwargs['splitter'] = splitter
                self._implementation = DenseRetriever(**init_kwargs)
            else:
                raise ValueError("Mode must be set to either 'sparse' or 'dense'")

        def retrieve(self, fact: str, article_texts_by_id: Dict[str, str], top_k: int = 5) -> List[Dict[str, Any]]:
            """
            ### Description

            Retrieve the ``top_k`` most relevant fragments across all articles in ``article_texts_by_id``.

            The articles are split into fragments according to ``splitter`` (set at
            initialization), then the selected backend retriever (sparse BM25 or
            dense embeddings) ranks those fragments against the provided ``fact``.

            ### Parameters

            :param fact: The query or factual statement to match.
            :type fact: str
            :param article_texts_by_id: Mapping of article_id -> full article text
            :type article_texts_by_id: Dict[str, str]
            :param top_k: Number of fragments to return. Capped by number of available fragments, defaults to ``5``.
            :type top_k: int, optional
            :returns: ``top_k`` fragments ordered from most to least relevant, each with ``text``, ``score``, and ``article_id``.
            :rtype: List[Dict[str, Any]]

            ### Notes

            - In sparse mode, BM25 runs over language-aware lemmas/tokens.
            - In dense mode, a HuggingFace encoder embeds the query and fragments;
              ranking is done via cosine similarity.
            """
            return self._implementation.retrieve(fact, article_texts_by_id, top_k=top_k)
    return Any, BM25Retriever, DenseRetriever, Dict, List, tqdm


@app.cell
def _(Any, BM25Retriever, DenseRetriever, Dict, List):
    class TwoStageRetriever:
        """
        Two-stage retriever.
        First stage: retrieve top-k full articles using sparse BM25.
        Second stage: retrieve top-k fragments from retrieved articles using dense embedder.
        """
    
        def __init__(self, 
                     bm25_retriever: BM25Retriever, 
                     dense_retriever: DenseRetriever,
                     corpus: Dict[str, str]):
            """
            Initialize TwoStageRetriever with pre-initialized retrievers.
        
            :param bm25_retriever: Pre-initialized BM25Retriever (should have splitter='article')
            :param dense_retriever: Pre-initialized DenseRetriever (should have splitter='sentence')
            :param corpus: Static corpus mapping article_id -> article_text
            """
            self.bm25_retriever = bm25_retriever
            self.dense_retriever = dense_retriever
            self.corpus = corpus
    
        def retrieve(self, 
                     fact: str, 
                     top_k_articles: int = 5, 
                     top_k_sentences: int = 5) -> List[Dict[str, Any]]:
            """
            Two-stage retrieval process.
        
            Stage 1: Use BM25 to retrieve top_k_articles from the static corpus.
            Stage 2: Combine retrieved articles and use DenseRetriever to get top_k_sentences.
        
            :param fact: Query/fact to search for
            :param top_k_articles: Number of articles to retrieve in stage 1
            :param top_k_sentences: Number of sentences to retrieve in stage 2
            :returns: List of top_k_sentences with 'text', 'score', 'article_id'
            """
            # Stage 1: Retrieve top-k articles using BM25
            stage1_results = self.bm25_retriever.retrieve(fact, article_texts_by_id={}, top_k=top_k_articles)
        
            if not stage1_results:
                return []
        
            # Extract unique article IDs from stage 1 results
            relevant_article_ids = list(set(result['article_id'] for result in stage1_results))
        
            # Build subcorpus with only the relevant articles
            subcorpus = {article_id: self.corpus[article_id] for article_id in relevant_article_ids}
        
            # Stage 2: Use DenseRetriever on the subcorpus to retrieve sentences
            stage2_results = self.dense_retriever.retrieve(fact, article_texts_by_id=subcorpus, top_k=top_k_sentences)
        
            return stage2_results
    return (TwoStageRetriever,)


@app.cell
def _(Dict):
    from datasets import load_dataset

    def load_hf_data(dataset_name: str, split: str='queries') -> Dict[str, str]:
        ds = load_dataset(dataset_name, split)
        ds_dict = {}
        for record in ds['train']:
            ds_dict[record['_id']] = record['text']
        return ds_dict
    return (load_hf_data,)


@app.cell
def _(BM25Retriever, DenseRetriever, TwoStageRetriever, load_hf_data, tqdm):
    import json
    import argparse

    def retrieve_from_corpus(corpus_name='kaengreg/wikifacts-articles'):
        corpus_dataset = load_hf_data(corpus_name, split='corpus')
        queries_dataset = load_hf_data(corpus_name, split='queries')
        corpus_dataset = dict(list(corpus_dataset.items())[:15])
        queries_dataset = dict(list(queries_dataset.items())[:3])
        stage_1_retriever = BM25Retriever(lang='ru', splitter='article', corpus=corpus_dataset)
        stage_2_retriever = DenseRetriever(model_name='intfloat/multilingual-e5-large', maxlen=256, pooling='mean', lang='ru', splitter='sentence', device='cpu')
        retriever = TwoStageRetriever(bm25_retriever=stage_1_retriever, dense_retriever=stage_2_retriever, corpus=corpus_dataset)
        retrieval_results = {}
        for qid, qtext in tqdm(queries_dataset.items(), desc='Retrieving relevant fragments from corpus'):  # For testing
            top = retriever.retrieve(qtext, top_k_articles=3, top_k_sentences=5)
            items = []
            for item in top:
                items.append({'text': item['text'], 'similarity': float(item['score']), 'article_id': item['article_id']})  # Initialize BM25 with pre-built corpus
            retrieval_results[qid] = items
        return retrieval_results  # Initialize dense retriever  # Initialize two-stage retriever
    return json, retrieve_from_corpus


@app.cell
def _(json, retrieve_from_corpus):
    retrieval_results = retrieve_from_corpus()

    out_path = f'retrieval_results_2_stage.jsonl'
    with open(out_path, 'w') as f:
        for qid, items in retrieval_results.items():
            rec = {
                'query_id': qid,
                'results': items,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote results to {out_path}")
    return


if __name__ == "__main__":
    app.run()

