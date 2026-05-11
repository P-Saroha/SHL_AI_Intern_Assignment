from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.catalog_loader import CatalogLoader
from app.config import settings
from app.models import CatalogItem
from app.utils import ensure_parent_dir, keyword_overlap_score


class HybridRetriever:
    """Hybrid retriever for the SHL assessment catalog.

    Beginner-friendly concepts:

    Embeddings:
        An embedding is a list of numbers that represents the meaning of text.
        We use `sentence-transformers/all-MiniLM-L6-v2` to convert each SHL
        assessment's searchable text into an embedding.

    Vector search:
        When a user asks for "Java developer assessment", we embed that query
        too, then search for catalog embeddings that are close to it.

    FAISS:
        FAISS is a fast vector search library. It stores the catalog embeddings
        and quickly returns the nearest vectors.

    Hybrid retrieval:
        Semantic search is good for meaning. Keyword overlap is good for exact
        terms like "HIPAA", "Excel", "Java", or "Spanish". We combine both:
        0.7 semantic similarity + 0.3 keyword overlap.

    Recall@10:
        The assignment scores whether relevant assessments appear in the top
        10 results. Good retrieval should avoid missing plausible relevant
        items, because the agent can only recommend what retrieval surfaces.
    """

    def __init__(
        self,
        catalog: list[CatalogItem] | None = None,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        embedding_model_name: str | None = None,
        model_cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.catalog = catalog or []
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.metadata_path = Path(metadata_path or settings.metadata_path)
        self.embedding_model_name = embedding_model_name or settings.embedding_model
        self.model_cache_dir = Path(model_cache_dir or settings.model_cache_dir)
        self.local_files_only = local_files_only

        self._model: Any | None = None
        self._index: Any | None = None
        self._metadata: list[dict[str, Any]] = []

    def build_and_save(self, catalog: list[CatalogItem] | None = None) -> None:
        """Build and save the FAISS index plus metadata.

        Run this offline or during deployment, not inside a user request.
        """
        if catalog is not None:
            self.catalog = catalog

        if not self.catalog:
            raise ValueError("Cannot build FAISS index because catalog is empty.")

        np = _import_numpy()
        faiss = _import_faiss()
        model = self._load_embedding_model()

        texts = [item.searchable_text for item in self.catalog]

        # Each catalog item becomes one vector. Normalizing embeddings lets us
        # use inner product search as cosine similarity.
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        ensure_parent_dir(self.index_path)
        ensure_parent_dir(self.metadata_path)
        faiss.write_index(index, str(self.index_path))

        metadata = [self._to_metadata(item) for item in self.catalog]
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._index = index
        self._metadata = metadata

    def load_index(self) -> None:
        """Load the saved FAISS index and metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            # On some deployment platforms, build artifacts may not persist.
            # Fall back to building the index at runtime to keep the API alive.
            loader = CatalogLoader(settings.catalog_path, settings.catalog_url)
            catalog = loader.load(download_if_missing=True)
            if not catalog:
                raise FileNotFoundError(
                    "Retrieval files are missing and catalog download failed. "
                    "Run: python -m app.retriever --build --force-download"
                )
            self.catalog = catalog
            self.build_and_save()

        faiss = _import_faiss()
        self._index = faiss.read_index(str(self.index_path))
        self._metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 10) -> list[dict[str, str]]:
        """Return top-k relevant assessments with name, url, and test_type."""
        if not query.strip():
            return []

        if self._index is None or not self._metadata:
            self.load_index()

        np = _import_numpy()
        model = self._load_embedding_model()

        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.asarray(query_embedding, dtype="float32")

        # The SHL catalog is small, so search the whole FAISS index before
        # hybrid reranking. This improves Recall@10 because exact keyword
        # matches are less likely to be accidentally filtered out too early.
        candidate_count = len(self._metadata)
        semantic_scores, candidate_ids = self._index.search(query_embedding, candidate_count)

        ranked: list[tuple[float, dict[str, Any]]] = []
        for semantic_score, item_id in zip(semantic_scores[0], candidate_ids[0]):
            if item_id < 0:
                continue

            item = self._metadata[item_id]

            # FAISS inner product is cosine similarity because embeddings are
            # normalized. Map roughly from -1..1 to 0..1 before blending.
            semantic = (float(semantic_score) + 1.0) / 2.0
            keyword = keyword_overlap_score(query, item["searchable_text"])
            final_score = (0.7 * semantic) + (0.3 * keyword)
            ranked.append((final_score, item))

        ranked.sort(key=lambda pair: pair[0], reverse=True)

        results: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for _, item in ranked:
            if item["url"] in seen_urls:
                continue
            seen_urls.add(item["url"])
            results.append(
                {
                    "name": item["name"],
                    "url": item["url"],
                    "test_type": item["test_type"],
                }
            )
            if len(results) >= top_k:
                break

        return results

    def _load_embedding_model(self) -> Any:
        """Load the sentence-transformer model lazily.

        The cache folder is inside the project so model files do not fill the
        default user cache on C:.
        """
        if self._model is None:
            SentenceTransformer = _import_sentence_transformer()
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self._model = SentenceTransformer(
                self.embedding_model_name,
                cache_folder=str(self.model_cache_dir),
                local_files_only=self.local_files_only,
            )
        return self._model

    @staticmethod
    def _to_metadata(item: CatalogItem) -> dict[str, Any]:
        """Convert a CatalogItem into JSON metadata saved beside FAISS."""
        return {
            "name": item.name,
            "url": str(item.url),
            "test_type": item.test_type,
            "description": item.description,
            "keys": item.keys,
            "job_levels": item.job_levels,
            "languages": item.languages,
            "duration": item.duration,
            "searchable_text": item.searchable_text,
        }


def build_retrieval_index(force_download: bool = False) -> None:
    """Download/load the SHL catalog and build retrieval artifacts."""
    loader = CatalogLoader(settings.catalog_path, settings.catalog_url)
    if force_download:
        loader.download(force=True)

    catalog = loader.load(download_if_missing=True)
    retriever = HybridRetriever(catalog=catalog)
    retriever.build_and_save()


def _import_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required. Install dependencies in myenv with: "
            ".\\myenv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc
    return SentenceTransformer


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("faiss-cpu is required. Install dependencies in myenv.") from exc
    return faiss


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required. Install dependencies in myenv.") from exc
    return np


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the SHL retrieval index.")
    parser.add_argument("--build", action="store_true", help="Build FAISS index and metadata.")
    parser.add_argument("--force-download", action="store_true", help="Download a fresh catalog first.")
    parser.add_argument("--query", type=str, default="", help="Run a retrieval query.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return.")
    args = parser.parse_args()

    if args.build:
        build_retrieval_index(force_download=args.force_download)
        print(f"Saved FAISS index: {settings.faiss_index_path}")
        print(f"Saved metadata: {settings.metadata_path}")

    if args.query:
        retriever = HybridRetriever(local_files_only=True)
        for result in retriever.search(args.query, top_k=args.top_k):
            print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
