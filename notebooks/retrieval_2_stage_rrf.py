import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import pickle
    from typing import List, Dict, Any, Optional

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity 
    from tqdm import tqdm

    import torch 
    import torch.nn.functional as F
    from datasets import load_dataset
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
    return (
        Any,
        AutoModel,
        AutoTokenizer,
        BM25,
        Dict,
        F,
        LANG_MAPPING,
        List,
        MultilingualLemmatizer,
        Optional,
        cosine_similarity,
        json,
        load_dataset,
        np,
        os,
        pickle,
        sent_tokenize,
        torch,
        tqdm,
    )


@app.cell
def _(
    Any,
    AutoModel,
    AutoTokenizer,
    Dict,
    F,
    LANG_MAPPING,
    List,
    cosine_similarity,
    np,
    sent_tokenize,
    torch,
    tqdm,
):
    class DenseRetriever:
        def __init__(self,
                model_name: str,
                maxlen: int, 
                pooling: str,
                splitter: str,
                lang: str,
                device: str = 'cuda',
                batch_size: int = 256,
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
            for i in range(0, len(texts), self.batch_size):
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
    return (DenseRetriever,)


@app.cell
def _(
    Any,
    BM25,
    Dict,
    LANG_MAPPING,
    List,
    MultilingualLemmatizer,
    Optional,
    os,
    pickle,
    sent_tokenize,
    tqdm,
):
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
            save_dir = '../data/vector_store/bm25'
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
            save_dir = '../data/vector_store/bm25'

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
    return (BM25Retriever,)


@app.cell
def _(Any, BM25Retriever, DenseRetriever, Dict, List):
    class TwoStageRetrieverRRF:
        """
        Two-stage retriever with Reciprocal Rank Fusion between stages.
    
        First stage: retrieve top-k1 full articles using sparse BM25.
        Second stage: retrieve top-k2 fragments from retrieved articles using dense embedder.

        After each stage, reciprocal ranks (weights) are assigned to the outputs.

        After end-goal documents are retrieved, RRF stage is performed on their weights.
        """

        def __init__(self, 
                     bm25_retriever: BM25Retriever, 
                     dense_retriever: DenseRetriever,
                     chunks: Dict[str, list[str]]):
            """
            Initialize TwoStageRetriever with pre-initialized retrievers.

            :param bm25_retriever: Pre-initialized BM25Retriever (should have splitter='article')
            :param dense_retriever: Pre-initialized DenseRetriever (should have splitter='sentence')
            :param chunks: Pre-split chunks for the second retrieval stage.  
            """
            self.bm25_retriever = bm25_retriever
            self.dense_retriever = dense_retriever
            self.chunks = chunks
            
        def _get_articles_len_penalties(self,
                                        articles: Dict[str, List[str]]) -> Dict[str, float]:
            """
            Calculate length penalties for long retrieved articles.
            Length is the sentences count.
            
            Penalty formula for article i:
            penalty_multiplier = max(0.7, 1 - ((len(i) - len(min_article) / ∑ len(j))))
            
            :param articles: Articles with their ids and sentences.
            :returns: Penalty multipliers for all articles.
            """
            penalties = {}
            
            # Fisrt run through to calculate min and sum
            min_len, sum_len = -1, 0
            for sentences in articles.values():
                cur_len = len(sentences)
                
                if cur_len < min_len or min_len == -1:
                    min_len = cur_len
                    
                sum_len += cur_len
                
            # Second run through to calculate penalty multipliers
            for idx in articles:
                penalties[idx] = max(0.7, 1 - ((len(articles[idx]) - min_len) / sum_len))
                
            return penalties
        
        def _get_sentence_len_penalty(self,
                                      sentence: str) -> float:
            """
            Calculate length penalty for the sentence.
            
            Numeric thresholds in penalty formula are deduced from qrels length distribution:
            1. 80 <= len <= 140 --> 1.0
            2. 60 <= len <= 155 --> 0.9
            3. 50 <= len <= 190 --> 0.8
            4. 35 <= len <= 230 --> 0.75
            5. otherwise --> 0.7
            
            :param sentence: Input sentence
            :returns: Penalty multiplier for the sentence
            """
            sent_len = len(sentence)
            
            if sent_len >= 80 and sent_len <= 140:
                return 1.0
            if sent_len >= 60 and sent_len <= 155:
                return 0.9
            if sent_len >= 50 and sent_len <= 190:
                return 0.8
            if sent_len >= 35 and sent_len <= 230:
                return 0.75
            
            return 0.7

        def _rerank_with_rrf(self,
                             docs: List[Dict[str, Any]],
                             weights: List[float] = [0.5, 0.5],
                             len_penalties: Dict[str, float] = None) -> List[Dict[str, Any]]:
            """
            Rerank retrieved documents using Weighted Reciprocal Rank Fusion.

            RRF formula for sentence with rank i and article with rank j:
            rrf = w0 * len_penalty[j] * (1 / j) + w1 * sentence_penalty[i] * (1 / i)

            :param docs: Retrieved documents with reciprocal ranks for both retrieval stages
            :param weights: Importance of each stage for RRF, sums up to 1
            :param len_penalties: Length penalties for long articles
            :returns: Reranked documents
            """
            # If length penalties were not provided, ignore them during computation
            if not len_penalties:
                len_penalties = {}
                for entry in docs:
                    if entry['article_id'] not in len_penalties:
                        len_penalties[entry['article_id']] = 1.0
            
            # Calculate RRF scores for all docs
            for entry in docs:
                rrf_score = (
                    weights[0] * len_penalties[entry['article_id']] * entry['article_reciprocal_rank'] 
                    + weights[1] * entry['sentence_penalty'] * entry['sentence_reciprocal_rank']
                )
                entry['rrf_score'] = rrf_score

            # Sort docs by RRF score decrease
            return sorted(docs, key=lambda x: x['rrf_score'], reverse=True)

        def retrieve(self, 
                     fact: str, 
                     top_k_articles: int = 3, 
                     top_k_sentences: int = 10,
                     stages_weights: List[float] = [0.5, 0.5]) -> List[Dict[str, Any]]:
            """
            Two-stage retrieval process.

            Stage 1: Use BM25 to retrieve top_k_articles from the static corpus.
            Stage 2: Combine retrieved articles and use DenseRetriever to get top_k_sentences.

            :param fact: Query/fact to search for
            :param top_k_articles: Number of articles to retrieve in stage 1
            :param top_k_sentences: Number of sentences to retrieve in stage 2
            :param stages_weights: Importance of each stage for relevance score, sums up to 1
            :returns: List of top_k_sentences with 'text', 'score', 'article_id'
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

                # Include only 1st mention of the article
                if entry_id not in article_to_rank:
                    article_to_rank[entry_id] = 1 / idx

            # Extract unique article IDs from stage 1 results
            relevant_article_ids = list(set(result['article_id'] for result in stage1_results))

            # Build subcorpus with only the chunks from relevant articles
            subcorpus = {}
            total_sentences = 0
            for article_id in relevant_article_ids:
                if article_id in self.chunks:
                    # Remove duplicate sentence inside article
                    article_sentences = list(set(self.chunks[article_id]))
                
                    subcorpus[article_id] = article_sentences
                    total_sentences += len(article_sentences)
                    
            # Calculate length penalties for all extracted articles
            len_penalties = self._get_articles_len_penalties(subcorpus)

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
                entry['sentence_penalty'] = self._get_sentence_len_penalty(entry['text'])

            # Rerank results using weighted RRF
            rrf_reranked_results = self._rerank_with_rrf(
                stage2_results, 
                weights=stages_weights,
                len_penalties=len_penalties,
            )

            return rrf_reranked_results[:top_k_sentences]
    return (TwoStageRetrieverRRF,)


@app.cell
def _(Dict, load_dataset):
    def load_hf_data(dataset_name: str, split: str='queries') -> Dict[str, str]:
        ds = load_dataset(dataset_name, split)
        ds_dict = {}
        for record in ds['train']:
            ds_dict[record['_id']] = record['text']
        return ds_dict
    return (load_hf_data,)


@app.cell
def _(
    BM25Retriever,
    DenseRetriever,
    TwoStageRetrieverRRF,
    json,
    load_hf_data,
    tqdm,
):
    def retrieve_from_corpus(
        top_k_articles,
        stages_weights,
        corpus_name='kaengreg/wikifacts-articles',
    ):
        corpus_dataset = load_hf_data(corpus_name, split="corpus")
        queries_dataset = load_hf_data(corpus_name, split="queries")

        # Load pre-split sentence chunks
        with open('../heavy_artifacts/articles_pre_split_sents.json', 'r') as f:
            chunks = json.load(f)

        # For testing
#         corpus_dataset = dict(list(corpus_dataset.items())[:15])
#         queries_dataset = dict(list(queries_dataset.items())[:3])

        # Initialize BM25 with pre-built corpus
        stage_1_retriever = BM25Retriever(
            lang='ru',
            splitter='article',
            corpus=corpus_dataset,
            reindex_corpus=False,
        )

        # Initialize dense retriever
        stage_2_retriever = DenseRetriever(
            model_name='intfloat/multilingual-e5-large',
            maxlen=256,
            pooling='mean',
            lang='ru',
            splitter='sentence',
            device='cuda:0',
        )

        # Initialize two-stage retriever
        retriever = TwoStageRetrieverRRF(
            bm25_retriever=stage_1_retriever,
            dense_retriever=stage_2_retriever,
            chunks=chunks,
        )

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
    return (retrieve_from_corpus,)


@app.cell
def _(json, retrieve_from_corpus):
    # Set hyperparameter grid
    top_k_articles_grid = [4]
    weights_grid = [[0.5, 0.5], [0.6, 0.4], [0.7, 0.3]]
    
    # Perform inference on grid
    for t_i in top_k_articles_grid:
        for w_i in weights_grid:
            print(f'Current grid value: top_k_articles = {t_i}, stages_weights = {w_i}')
            retrieval_results = retrieve_from_corpus(t_i, w_i)

            out_path = f'../heavy_artifacts/with_both_penalties/retrieval_results_2_stage_pre_split_{t_i}_{w_i[0]}_{w_i[1]}.jsonl'
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
