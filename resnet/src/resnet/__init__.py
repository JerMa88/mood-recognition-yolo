"""resnet package for facial expression recognition (FER)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("landmark-embedding")
except PackageNotFoundError:
    __version__ = "0.1.0"

def ping() -> str:
    return "resnet"

__all__ = ["__version__", "ping"]

