import random
import os

from qdrant_client import QdrantClient, models
from data_loader import load_queries, load_corpus
from fastembed import SparseTextEmbedding

from nltk.tokenize import sent_tokenize

from tqdm import tqdm


def prepare_test_data(dataset_name, lang):
    """Loads data and selects a single random query with relevant articles."""
    queries = load_queries(dataset_name, f"{lang}_queries")
    corpus = load_corpus(dataset_name, f"{lang}_corpus")

    qids = list(queries.keys())
    random.shuffle(qids)

    record = None
    for qid in qids:
        record = queries[qid]
        if record.get("relevant articles"):
            article_ids = record.get("relevant articles", [])
            relevant_corpus_check = {aid: corpus[aid] for aid in article_ids if aid in corpus}
            if relevant_corpus_check:
                break
    
    if not record or not record.get("relevant articles"):
        raise RuntimeError("Could not find any queries with relevant articles in the dataset.")

    fact = record['text']
    article_ids = record.get("relevant articles", [])
    relevant_corpus = {aid: corpus[aid] for aid in article_ids if aid in corpus}
    
    return fact, relevant_corpus


class QdrantRetriever:
    def __init__(self, collection_name="wikifacts-bench", model_name="prithivida/Splade_PP_en_v1", splitter='sentence'):
        self.client = QdrantClient(":memory:")
        self.collection_name = collection_name
        self.model = SparseTextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.splitter = splitter
        
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={},
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            },
        )

    def split_sentence(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def split_abstract(self, text: str) -> list[str]:
        return [para.strip() for para in text.split("\n\n") if para.strip() != ""]

    def split(self, text: str) -> list[str]:
        if self.splitter == "sentence":
            return self.split_sentence(text)
        return self.split_abstract(text)

    def index_corpus(self, corpus):
        points = []
        point_id = 0
        for doc_id, doc in corpus.items():
            fragments = self.split(doc['text'])
            embeddings = list(self.model.embed(fragments))

            for i, fragment in enumerate(fragments):
                points.append(
                    models.PointStruct(
                        id=point_id,
                        payload={"doc_id": doc_id, "text": fragment},
                        vector={
                            "text-sparse": models.SparseVector(
                                indices=embeddings[i].indices.tolist(),
                                values=embeddings[i].values.tolist(),
                            )
                        },
                    )
                )
                point_id += 1
        
        if not points:
            return

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
        )
    
    def search(self, query, top_k=5):
        query_embedding = list(self.model.embed([query]))[0]

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name="text-sparse",
                vector=models.SparseVector(
                    indices=query_embedding.indices.tolist(),
                    values=query_embedding.values.tolist(),
                ),
            ),
            limit=top_k,
        )
        return hits

    def run_test(self, fact, relevant_corpus, lang):
        if not relevant_corpus:
            print(f"No relevant articles found for query: {fact}")
            return
            
        print(f"Indexing {len(relevant_corpus)} relevant articles for the query...")
        self.index_corpus(relevant_corpus)
        
        retrieved_docs = self.search(fact)
        
        os.makedirs("test", exist_ok=True)
        sanitized_model_name = self.model_name.replace("/", "_")

        with open(f"test/{sanitized_model_name}__{self.splitter}__{lang}.txt", "w") as f:
            f.write(f"Query: {fact}\n")
            f.write("\nRetrieved Documents:\n")

            for i, doc in enumerate(retrieved_docs, 1):
                f.write(f"{i}. {doc.payload['text']} (Score: {doc.score})\n")

if __name__ == "__main__":
    random.seed(5)
    
    DATASET_NAME = "kaengreg/wikifacts-bench"
    LANG = "ru"

    fact, relevant_corpus = prepare_test_data(DATASET_NAME, LANG)

    sparse_model_list = [
        'prithivida/Splade_PP_en_v1',
        'Qdrant/bm42-all-minilm-l6-v2-attentions',
        'Qdrant/bm25',
        'Qdrant/minicoil-v1',
    ]
    splitters = [
        'sentence',
        'paragraph',
    ]

    for model_name in tqdm(sparse_model_list):
        for splitter in splitters:
            retriever = QdrantRetriever(splitter=splitter, model_name=model_name)
            retriever.run_test(fact, relevant_corpus, LANG)
