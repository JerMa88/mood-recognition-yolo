"""landmark_embedding package.

Minimal scaffold for development and testing.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # best-effort version when installed; fallback for local dev
    __version__ = version("landmark-embedding")
except PackageNotFoundError:  # package not installed yet
    __version__ = "0.1.0"


def ping() -> str:
    """Lightweight health check used by tests."""
    return "landmark_embedding"


__all__ = ["__version__", "ping"]

