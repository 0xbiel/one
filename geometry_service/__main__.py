"""Run the service with ``python -m geometry_service``."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("ONE_GEOMETRY_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("ONE_GEOMETRY_PORT", "8090"))
    except ValueError:
        port = 8090
    uvicorn.run("geometry_service.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
