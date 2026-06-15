"""LOKMA exception hierarchy.

A single root (``LokmaError``) lets callers catch any domain failure while still
allowing precise handling of specific conditions.
"""

from __future__ import annotations


class LokmaError(Exception):
    """Base class for all LOKMA domain errors."""


class ConfigurationError(LokmaError):
    """Raised when configuration is missing or invalid."""


class ModelLoadError(LokmaError):
    """Raised when the YOLO segmentation model cannot be loaded."""


class DatabaseError(LokmaError):
    """Raised on knowledge-base connection or query failures."""


class KnowledgeBaseBuildError(LokmaError):
    """Raised when the offline knowledge-base build fails."""


class CalibrationError(LokmaError):
    """Raised when camera calibration parameters are invalid."""
