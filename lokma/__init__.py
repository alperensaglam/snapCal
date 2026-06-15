"""LOKMA — optical nutrition scale.

Reconstructs food mass from a camera frame via ``m = V × ρ``:
recovered real-world volume ``V`` paired with a hierarchical density ``ρ``
from the USDA / Nutrition5k knowledge base.

Phase 1 establishes the service architecture:
``DatabaseManager``, ``DensityService``, ``VolumeEngineService``,
``CalibrationService`` and ``InferencePipeline``.
"""

__version__ = "0.1.0"
