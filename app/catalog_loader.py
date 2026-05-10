import json
from pathlib import Path
from urllib.request import urlopen

from app.config import settings
from app.models import CatalogItem
from app.utils import ensure_parent_dir, normalize_text


KEY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


class CatalogLoader:
    """Downloads, loads, and normalizes SHL catalog data.

    The raw SHL JSON has many fields. The rest of the app should not need to
    understand that raw shape, so this class converts each row into a clean
    `CatalogItem`.
    """

    def __init__(self, catalog_path: str, catalog_url: str | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog_url = catalog_url or settings.catalog_url

    def download(self, force: bool = False) -> Path:
        """Download the catalog once and cache it locally.

        Production note:
        User requests should not fetch the catalog from the internet. Download
        during setup/deploy, then serve from the local JSON file.
        """
        if self.catalog_path.exists() and not force:
            return self.catalog_path

        ensure_parent_dir(self.catalog_path)
        with urlopen(self.catalog_url, timeout=45) as response:
            self.catalog_path.write_bytes(response.read())

        return self.catalog_path

    def load(self, download_if_missing: bool = False) -> list[CatalogItem]:
        """Load catalog items from disk.

        If `download_if_missing=True`, this method downloads the SHL JSON first
        when the local file is absent.
        """
        if not self.catalog_path.exists() and download_if_missing:
            self.download()

        if not self.catalog_path.exists():
            return []

        raw_items = json.JSONDecoder(strict=False).decode(
            self.catalog_path.read_text(encoding="utf-8")
        )

        catalog: list[CatalogItem] = []
        for raw in raw_items:
            if raw.get("status") != "ok":
                continue

            keys = raw.get("keys") or []
            test_type = ",".join(KEY_TO_CODE[key] for key in keys if key in KEY_TO_CODE)

            catalog.append(
                CatalogItem(
                    name=raw.get("name", ""),
                    url=raw.get("link", ""),
                    description=raw.get("description", "") or "",
                    test_type=test_type,
                    keys=keys,
                    job_levels=raw.get("job_levels") or [],
                    languages=raw.get("languages") or [],
                    duration=raw.get("duration") or "",
                    searchable_text=self.build_searchable_text(raw),
                )
            )

        return catalog

    @staticmethod
    def build_searchable_text(raw: dict) -> str:
        """Create the exact text used for search and embeddings.

        We include:
        - name: captures product names like "Java 8" or "OPQ32r"
        - description: captures what the test measures
        - keys: captures broad categories like Knowledge & Skills
        - job_levels: helps match graduate, manager, executive, etc.
        - languages: helps match language-specific constraints
        """
        parts = [
            raw.get("name", ""),
            raw.get("description", "") or "",
            " ".join(raw.get("keys") or []),
            " ".join(raw.get("job_levels") or []),
            " ".join(raw.get("languages") or []),
        ]
        return normalize_text(" ".join(parts))
