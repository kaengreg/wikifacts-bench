from typing import List, Dict, Any

from tqdm import tqdm

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity 

import torch 
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

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
        assert splitter in ("sentence", "paragraph"), ""

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
        return self.split_paragraph(text)

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
    def __init__(self,
            lang: str,
            splitter: str,
            k1: float = 1.2,
            b: float = 0.75,
    ):
        assert splitter in ("sentence", "paragraph"), ""

        self.splitter = splitter
        self.lang = lang
        self.tokenizer = MultilingualLemmatizer(lang)
        self.k1 = k1
        self.b = b

    def split_sentence(self, text: str) -> list[str]:
        return [sent.strip() for sent in sent_tokenize(text, language=LANG_MAPPING[self.lang])]

    def split_paragraph(self, text: str) -> list[str]:
        return [para.strip() for para in text.split("\n\n") if para.strip() != ""]

    def split(self, text: str) -> list[str]:
        if self.splitter == "sentence":
            return self.split_sentence(text)
        return self.split_paragraph(text)

    def _tokenize(self, text: str) -> List[str]:
        # Use lemmatizer to normalize; then whitespace split to tokens
        normalized = self.tokenizer.lemmatize_text(text, remove_stopwords=True)
        return [tok for tok in normalized.split() if tok]

    def retrieve(self, fact: str, article_texts_by_id: Dict[str, str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top fragments across multiple articles using BM25.

        Returns a list of dicts: {'text': str, 'score': float, 'article_id': str}
        """
        fragments: List[str] = []
        owners: List[str] = []
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
    :param splitter: Granularity for splitting article text (``'sentence'`` or ``'paragraph'``), defaults to ``'sentence'``.
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
