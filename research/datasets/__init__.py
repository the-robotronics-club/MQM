"""Manifest-based research datasets and session-grouped splits."""

from .loader import load_manifest
from .schema import DatasetManifest, ImageRecord, PersonAnnotation

__all__ = ["DatasetManifest", "ImageRecord", "PersonAnnotation", "load_manifest"]
