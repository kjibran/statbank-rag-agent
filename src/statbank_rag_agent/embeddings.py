from pathlib import Path

from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
CACHE_DIR = str(Path.home() / ".cache" / "fastembed")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = TextEmbedding(MODEL_NAME, cache_dir=CACHE_DIR)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embeddings for stored search texts."""
    return [vector.tolist() for vector in _get_model().embed(texts, batch_size=64)]


def embed_query(question: str) -> list[float]:
    """Embedding for a search query. This model family expects an instruction prefix on queries."""
    return next(iter(_get_model().embed([QUERY_PREFIX + question]))).tolist()


def to_pgvector(vector: list[float]) -> str:
    """pgvector accepts vectors written as text, like '[0.1,0.2,0.3]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"
