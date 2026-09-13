import json
from pathlib import Path
from typing import Any


class LocalObjectStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        """Resolve a storage key without allowing it to escape the store root."""
        if not key or "\\x00" in key:
            raise ValueError("object key is invalid")
        candidate = Path(key)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("object key must stay inside the object store")
        root = self.root.resolve()
        destination = (root / candidate).resolve()
        try:
            destination.relative_to(root)
        except ValueError as exc:
            raise ValueError("object key must stay inside the object store") from exc
        return destination

    def put_json(self, key: str, value: dict[str, Any]) -> str:
        destination = self._path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
        return key

    def put_bytes(self, key: str, value: bytes) -> str:
        destination = self._path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        return key

    def get_bytes(self, key: str) -> bytes:
        return self._path_for(key).read_bytes()

    def delete(self, key: str) -> None:
        path = self._path_for(key)
        if path.exists(): path.unlink()
