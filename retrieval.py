from transformers import AutoModel, AutoTokenizer
import torch 
from nltk import sent_tokenize
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
from sklearn.metrics.pairwise import cosine_similarity 

class RelevantRetriever:
    def __init__(self,
            model_name: str,
            maxlen: str, 
            batch_size: int,
            pooling: str,
            splitter: str,
            device: str = 'cuda'
    ): 
        assert pooling in ("mean", "cls"), "pooling must be either mean or cls"
        assert splitter in ("sentence", "paragraph"), ""

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = self.load_model(model_name, device)

        self.maxlen = maxlen
        self.batch_size = batch_size
        self.pooling = pooling.lower()

    def load_model(self, model_name: str, device: str = 'cuda'):
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def split_sentence(self, text: str) -> list[str]:
        return sent_tokenize(text)

    def split_abstract(self, text: str) -> list[str]:
        return [para.strip() for para in text.split("\n\n") if para.strip() != ""]

    def split(self, text: str) -> list[str]:
        if self.splitter == "sentence":
            return self._split_sentences(text)
        return self._split_paragraphs(text)

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + self.batch_size]
            batch_dict = self.tokenizer(batch_texts, max_length=self.max_len, padding=True, truncation=True,
                                        return_tensors='pt')
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
            if self.pooling == 'average':
                batch_embeddings = self._average_pool(outputs, batch_dict['attention_mask'])
            elif self.pooling == 'cls':
                batch_embeddings = self._cls_pool(outputs)
            else:
                raise ValueError(f"Unknown pooling method: {self.pooling}")

            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)
    
    def retrieve(self, fact: str, article_text: str, top_k: int = 5) -> list[str]:
        fragments = self.split(article_text)
        if not fragments:
            return []

        query_emb = self.get_embeddings([fact])
        frag_embs = self.get_embeddings(fragments)

        sims = cosine_similarity(query_emb, frag_embs)[0]
        top_idx = sims.argsort()[::-1][:top_k]

        return [fragments[i] for i in top_idx]
