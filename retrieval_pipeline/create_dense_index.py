"""Build a dense vector index for a WikiFacts split.

The dataset load, deduplication, and output files match the original E5
script: each split is written as ``embeddings.npy`` plus ``metadata.json``.
Metadata stores the raw texts. Prefixes are applied only while embedding.

Default ``--model e5`` reproduces the previous run. Other profiles change
the checkpoint, prefixes, pooling, and inference backend.

Transformers is the default for the encoder models. Giga and Qwen3 default
to vLLM and expect that package in a separate environment. Pass
``--backend transformers`` to run them with Hugging Face Transformers.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_NAME = "kaengreg/wikifacts-para"
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "vector_store")

GIGA_QUERY_PREFIX = "Instruct: Given a query, retrieve relevant passages\nQuery: "
QWEN_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


@dataclass(frozen=True)
class EmbedderProfile:
    key: str
    model_name: str
    backend: str
    pooling: str
    max_len: int
    batch_size: int
    query_prefix: str
    document_prefix: str
    loader: str = "auto"
    trust_remote_code: bool = False
    torch_dtype: str | None = None
    add_pooling_layer: bool | None = None
    padding_side: str | None = None
    attn_implementation: str | None = None
    vllm_supported: bool = True
    vllm_chunk_size: int = 256
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)


def _mean_pooler(normalize: bool = True) -> dict[str, Any]:
    return {"pooling_type": "MEAN", "use_activation": normalize}


def _cls_pooler(normalize: bool = True) -> dict[str, Any]:
    return {"pooling_type": "CLS", "use_activation": normalize}


def _last_pooler(normalize: bool = True) -> dict[str, Any]:
    return {"pooling_type": "LAST", "use_activation": normalize}


PROFILES: dict[str, EmbedderProfile] = {
    "e5": EmbedderProfile(
        key="e5",
        model_name="intfloat/multilingual-e5-large",
        backend="transformers",
        pooling="mean",
        max_len=512,
        batch_size=64,
        query_prefix="query: ",
        document_prefix="passage: ",
        vllm_kwargs={
            "runner": "pooling",
            "pooler_config": _mean_pooler(),
        },
    ),
    "frida": EmbedderProfile(
        key="frida",
        model_name="ai-forever/FRIDA",
        backend="transformers",
        pooling="cls",
        max_len=512,
        batch_size=32,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        loader="t5_encoder",
        vllm_supported=False,
    ),
    "arctic-l-v2": EmbedderProfile(
        key="arctic-l-v2",
        model_name="Snowflake/snowflake-arctic-embed-l-v2.0",
        backend="transformers",
        pooling="cls",
        max_len=8192,
        batch_size=32,
        query_prefix="query: ",
        document_prefix="",
        add_pooling_layer=False,
        vllm_kwargs={
            "runner": "pooling",
            "pooler_config": _cls_pooler(),
        },
    ),
    "giga-3b": EmbedderProfile(
        key="giga-3b",
        model_name="ai-sage/Giga-Embeddings-instruct-3B-0826",
        backend="vllm",
        pooling="mean",
        max_len=2048,
        batch_size=8,
        query_prefix=GIGA_QUERY_PREFIX,
        document_prefix="",
        trust_remote_code=True,
        torch_dtype="bfloat16",
        vllm_chunk_size=128,
        vllm_kwargs={
            "runner": "pooling",
            "convert": "embed",
            "hf_overrides": {
                "is_causal": False,
                "architectures": ["Qwen3ForCausalLM"],
            },
            "pooler_config": _mean_pooler(),
            "trust_remote_code": True,
        },
    ),
    "qwen3-4b": EmbedderProfile(
        key="qwen3-4b",
        model_name="Qwen/Qwen3-Embedding-4B",
        backend="vllm",
        pooling="last",
        max_len=8192,
        batch_size=4,
        query_prefix=QWEN_QUERY_PREFIX,
        document_prefix="",
        torch_dtype="float16",
        padding_side="left",
        attn_implementation="flash_attention_2",
        vllm_chunk_size=128,
        vllm_kwargs={
            "runner": "pooling",
            "convert": "embed",
            "pooler_config": _last_pooler(),
        },
    ),
}


def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean":
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    if method == "cls":
        return hidden[:, 0]
    if method == "last":
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return hidden[:, -1]
        positions = attention_mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
    raise ValueError(f"Unknown pooling method: {method}")


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.size == 0:
        return embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def resolve_dtype(name: str | None, device: str) -> torch.dtype | None:
    if name is None:
        return None
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    if device == "cpu" and dtype == torch.float16:
        print("float16 is unavailable on CPU. Loading the model in float32.")
        return torch.float32
    return dtype


def load_transformers_model(profile: EmbedderProfile, dtype: torch.dtype | None):
    kwargs: dict[str, Any] = {}
    if profile.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    if profile.add_pooling_layer is not None:
        kwargs["add_pooling_layer"] = profile.add_pooling_layer
    if profile.attn_implementation:
        kwargs["attn_implementation"] = profile.attn_implementation

    if profile.loader == "t5_encoder":
        from transformers import T5EncoderModel

        model_cls = T5EncoderModel
    elif profile.loader == "auto":
        model_cls = AutoModel
    else:
        raise ValueError(f"Unknown loader: {profile.loader}")

    try:
        return _from_pretrained(model_cls, profile.model_name, kwargs)
    except Exception as exc:
        if "attn_implementation" not in kwargs:
            raise
        print(f"Could not enable {profile.attn_implementation} ({exc}). Loading without it.")
        kwargs.pop("attn_implementation")
        return _from_pretrained(model_cls, profile.model_name, kwargs)


def _from_pretrained(model_cls, model_name: str, kwargs: dict[str, Any]):
    try:
        return model_cls.from_pretrained(model_name, **kwargs)
    except TypeError:
        if "torch_dtype" not in kwargs:
            raise
        renamed = dict(kwargs)
        renamed["dtype"] = renamed.pop("torch_dtype")
        return model_cls.from_pretrained(model_name, **renamed)


class TransformersEmbedder:
    def __init__(self, profile: EmbedderProfile, device: str, batch_size: int, max_len: int):
        self.profile = profile
        self.device = device
        self.batch_size = batch_size
        self.max_len = max_len

        tokenizer_kwargs: dict[str, Any] = {}
        if profile.trust_remote_code:
            tokenizer_kwargs["trust_remote_code"] = True
        if profile.padding_side:
            tokenizer_kwargs["padding_side"] = profile.padding_side
        self.tokenizer = AutoTokenizer.from_pretrained(profile.model_name, **tokenizer_kwargs)

        dtype = resolve_dtype(profile.torch_dtype, device)
        self.model = load_transformers_model(profile, dtype).to(device).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        batches = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch_texts = texts[start : start + self.batch_size]
            batch = self.tokenizer(
                batch_texts,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch = {key: value.to(self.device) for key, value in batch.items()}
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
                pooled = pool_hidden(hidden, batch["attention_mask"], self.profile.pooling)
                pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.float().cpu().numpy())
        return l2_normalize(np.vstack(batches).astype(np.float32))


def _pooler_config(spec: dict[str, Any]):
    from vllm.config import PoolerConfig

    try:
        return PoolerConfig(**spec)
    except TypeError:
        if "use_activation" not in spec:
            raise
        legacy = dict(spec)
        legacy["normalize"] = legacy.pop("use_activation")
        return PoolerConfig(**legacy)


def _build_vllm(model_name: str, kwargs: dict[str, Any]):
    from vllm import LLM

    prepared = dict(kwargs)
    if "pooler_config" in prepared:
        prepared["pooler_config"] = _pooler_config(prepared["pooler_config"])
    try:
        return LLM(model=model_name, **prepared)
    except TypeError as exc:
        if "runner" not in prepared and "convert" not in prepared:
            raise
        legacy = {key: value for key, value in prepared.items() if key not in {"runner", "convert"}}
        legacy["task"] = "embed"
        print(f"Retrying vLLM load with task='embed' after: {exc}")
        return LLM(model=model_name, **legacy)


def _embedding_vector(output) -> np.ndarray:
    data = getattr(output.outputs, "embedding", None)
    if data is None:
        data = output.outputs.data
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    vector = np.asarray(data, dtype=np.float32)
    if vector.ndim != 1:
        raise RuntimeError(
            f"Expected one embedding vector per text, got shape {vector.shape}. "
            "The vLLM pooler did not reduce the sequence."
        )
    return vector


class VllmEmbedder:
    def __init__(self, profile: EmbedderProfile, max_len: int, chunk_size: int):
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "vLLM is not installed. Giga and Qwen3 use it by default. "
                "Install vLLM in a separate environment, or rerun with --backend transformers."
            ) from exc

        kwargs = dict(profile.vllm_kwargs)
        kwargs.setdefault("max_model_len", max_len)
        if profile.trust_remote_code:
            kwargs.setdefault("trust_remote_code", True)
        print(f"Loading vLLM embedder for {profile.model_name}...")
        self.llm = _build_vllm(profile.model_name, kwargs)
        self.chunk_size = chunk_size
        self.max_len = max_len

    def _encode(self, texts: list[str]):
        if hasattr(self.llm, "embed"):
            try:
                return self.llm.embed(texts, use_tqdm=False)
            except TypeError:
                return self.llm.embed(texts)
        try:
            return self.llm.encode(texts, pooling_task="embed", use_tqdm=False)
        except TypeError:
            return self.llm.encode(texts, pooling_task="embed")

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        batches = []
        for start in tqdm(range(0, len(texts), self.chunk_size), desc="Embedding"):
            outputs = self._encode(texts[start : start + self.chunk_size])
            batches.append(np.stack([_embedding_vector(output) for output in outputs]))
        return l2_normalize(np.vstack(batches).astype(np.float32))


def build_embedder(
    profile: EmbedderProfile,
    backend: str,
    device: str,
    batch_size: int,
    max_len: int,
    vllm_chunk_size: int,
):
    if backend == "transformers":
        print(f"Loading {profile.model_name} with Transformers on {device}...")
        return TransformersEmbedder(profile, device, batch_size, max_len)
    if backend == "vllm":
        if not profile.vllm_supported:
            raise SystemExit(
                f"{profile.key} is a T5 encoder. vLLM's embedding list does not include "
                "T5EncoderModel. Run it with --backend transformers."
            )
        return VllmEmbedder(profile, max_len, vllm_chunk_size)
    raise ValueError(f"Unknown backend: {backend}")


def load_hf_texts(dataset_name: str, split: str) -> list[str]:
    """Load a WikiFacts config named ``split`` and return its texts.

    The second positional argument of ``load_dataset`` is the config name.
    These repos expose ``corpus`` and ``queries`` as configs, each with a
    ``train`` split.
    """
    print(f"Loading dataset {dataset_name} (config: {split})...")
    ds = load_dataset(dataset_name, split)
    if hasattr(ds, "keys") and "train" in ds:
        data = ds["train"]
    else:
        data = ds

    texts = []
    for record in tqdm(data, desc=f"Parsing {split}"):
        texts.append(record["text"])
    return texts


def prefix_for_split(profile: EmbedderProfile, split_name: str) -> str:
    if split_name == "queries":
        return profile.query_prefix
    return profile.document_prefix


def process_and_save(
    split_name: str,
    profile: EmbedderProfile,
    embedder,
    dataset_name: str,
    output_dir: str,
    deduplicate: bool,
) -> None:
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    texts = load_hf_texts(dataset_name, split_name)
    if deduplicate:
        print(f"Deduplicating {split_name}...")
        original_count = len(texts)
        texts = sorted(set(texts))
        print(f"Reduced {split_name} from {original_count} to {len(texts)} unique texts.")

    prefix = prefix_for_split(profile, split_name)
    prepared = [f"{prefix}{text}" for text in texts]
    print(f"Generating embeddings for {split_name} ({len(prepared)} texts)...")
    embeddings = embedder.embed(prepared)

    emb_path = os.path.join(split_dir, "embeddings.npy")
    meta_path = os.path.join(split_dir, "metadata.json")
    profile_path = os.path.join(split_dir, "profile.json")

    print(f"Saving embeddings to {emb_path}...")
    np.save(emb_path, embeddings)
    print(f"Saving metadata to {meta_path}...")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(texts, handle, ensure_ascii=False, indent=4)
    with open(profile_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_key": profile.key,
                "model_name": profile.model_name,
                "pooling": profile.pooling,
                "max_len": embedder.max_len if hasattr(embedder, "max_len") else profile.max_len,
                "query_prefix": profile.query_prefix,
                "document_prefix": profile.document_prefix,
                "split": split_name,
                "prefix": prefix,
                "normalized": True,
            },
            handle,
            ensure_ascii=False,
            indent=4,
        )
    print(f"Saved {split_name} index to {split_dir}")


def resolve_output_dir(raw: str | None, profile_key: str) -> str:
    """Resolve an index directory from the repository root."""
    if raw is None:
        return os.path.join(BASE_OUTPUT_DIR, profile_key)
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(REPO_ROOT, raw))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dense embedding index for a WikiFacts dataset config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=sorted(PROFILES), default="e5", help="Embedder profile.")
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default=None,
        help="Override the profile backend. FRIDA stays on Transformers.",
    )
    parser.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset repo.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Index directory, relative to the repository root. Defaults to data/vector_store/<model key>.",
    )
    parser.add_argument("--device", default=None, help="Transformers device, for example cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Transformers batch size, or the vLLM chunk size when that backend is selected.",
    )
    parser.add_argument("--max-len", type=int, default=None, help="Maximum tokens per text.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["queries", "corpus"],
        help="Dataset configs to embed. The corpus config is deduplicated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.model]
    backend = args.backend or profile.backend
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or profile.batch_size
    vllm_chunk_size = args.batch_size or profile.vllm_chunk_size
    max_len = args.max_len or profile.max_len
    output_dir = resolve_output_dir(args.output_dir, profile.key)

    print(
        f"Model {profile.key} ({profile.model_name}), backend {backend}, "
        f"pooling {profile.pooling}, max_len {max_len}."
    )
    if backend == "vllm" and args.device:
        print("vLLM selects the GPU itself. Set CUDA_VISIBLE_DEVICES before launching; --device is ignored.")
    embedder = build_embedder(profile, backend, device, batch_size, max_len, vllm_chunk_size)
    for split_name in args.splits:
        process_and_save(
            split_name=split_name,
            profile=profile,
            embedder=embedder,
            dataset_name=args.dataset,
            output_dir=output_dir,
            deduplicate=split_name == "corpus",
        )
    print("All tasks completed!")


if __name__ == "__main__":
    main()
