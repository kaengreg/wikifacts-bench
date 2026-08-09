import os
import pickle
from typing import List, Dict, Any, Optional

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

# WIP: languages that have their own nltk tokenizer
LANG_MAPPING = {
    'ru': 'russian',
    'en': 'english',
}


# os.environ['HF_DATASETS_CACHE'] = '~/hf_cache'


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
            corpus_dir: str,
            corpus: Optional[Dict[str, str]] = None,
            reindex_corpus: bool = True,
            k1: float = 1.2,
            b: float = 0.75,
    ):
        assert splitter in ("sentence", "paragraph", "article"), ""

        self.splitter = splitter
        self.lang = lang
        self.corpus = corpus
        self.corpus_dir = corpus_dir
        self.tokenizer = MultilingualLemmatizer(lang)
        self.k1 = k1
        self.b = b

        self._index = []
        self._texts = []
        self._owners = []
        self.model = None

        if corpus:
            if reindex_corpus:
                self.create_index(self.corpus)
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
        save_dir = f'../data/vector_store/bm25/{self.corpus_dir}'
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
        save_dir = f'../data/vector_store/bm25/{self.corpus_dir}'

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


def load_hf_data(dataset_name: str, split: str='queries') -> Dict[str, str]:
    ds = load_dataset(dataset_name, split)
    ds_dict = {}
    for record in ds['train']:
        ds_dict[record['_id']] = record['text']
    return ds_dict


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-stage retrieval with RRF and penalties")
    parser.add_argument("--corpus_name", type=str, default="kaengreg/wikifacts-para", help="HF dataset name")
    args = parser.parse_args()

    # 1. Load data
    print("Loading datasets...")
    corpus_dataset = load_hf_data(args.corpus_name, split="corpus")
    queries_dataset = load_hf_data(args.corpus_name, split="queries")

    # 4. Initialize retriever and create index
    print("Initializing retrievers...")
    bm25_retriever = BM25Retriever(
        lang='ru',
        splitter='article',
        corpus=corpus_dataset,
        corpus_dir='para',
        reindex_corpus=True,
    )


if __name__ == "__main__":
    main()
