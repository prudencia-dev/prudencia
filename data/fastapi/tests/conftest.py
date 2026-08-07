"""Configuration légère des tests unitaires hors conteneur Docker."""

from __future__ import annotations

import sys
from types import ModuleType

try:
    import chromadb  # noqa: F401
except ModuleNotFoundError:
    chromadb_stub = ModuleType("chromadb")

    def unavailable_http_client(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("ChromaDB n'est pas installé dans cet environnement.")

    chromadb_stub.HttpClient = unavailable_http_client
    sys.modules["chromadb"] = chromadb_stub
