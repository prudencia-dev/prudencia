from typing import Any

import pytest
from app.ai import rag
from app.ai.embedding_config import RAG_EMBEDDING_MODEL


class FakeCollection:
    def __init__(self, metadata: dict[str, Any] | None = None, count: int = 0):
        self.metadata = metadata or {}
        self._count = count
        self.modified_metadata: dict[str, Any] | None = None

    def count(self) -> int:
        return self._count

    def modify(self, *, metadata: dict[str, Any]) -> None:
        self.metadata = metadata
        self.modified_metadata = metadata

    def upsert(self, **kwargs: Any) -> None:
        self._count += len(kwargs["ids"])


class FakeClient:
    def __init__(self, collection: FakeCollection):
        self.collection = collection

    def get_or_create_collection(self, **kwargs: Any) -> FakeCollection:
        return self.collection


def test_empty_collection_receives_bge_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(metadata={"hnsw:space": "cosine"})
    monkeypatch.setattr(rag, "get_chroma_client", lambda: FakeClient(collection))

    rag.get_or_create_collection("documents", embedding_dimension=1024)

    assert collection.modified_metadata == {
        "hnsw:space": "cosine",
        "embedding_model": RAG_EMBEDDING_MODEL,
        "embedding_normalized": True,
        "embedding_dimension": 1024,
    }


def test_legacy_non_empty_collection_must_be_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = FakeCollection(metadata={"hnsw:space": "cosine"}, count=3)
    monkeypatch.setattr(rag, "get_chroma_client", lambda: FakeClient(collection))

    with pytest.raises(ValueError, match="ne déclare pas son modèle"):
        rag.get_or_create_collection("documents", embedding_dimension=1024)


def test_mixed_embedding_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="même dimension"):
        rag.add_chunks(
            collection_name="documents",
            ids=["1", "2"],
            texts=["a", "b"],
            embeddings=[[0.1, 0.2], [0.1]],
            metadatas=[{}, {}],
        )
