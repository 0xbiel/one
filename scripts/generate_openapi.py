"""Regenerate the committed API contract used by ONE clients."""

import json
from pathlib import Path

from app.main import app


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "contracts" / "openapi.json"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
