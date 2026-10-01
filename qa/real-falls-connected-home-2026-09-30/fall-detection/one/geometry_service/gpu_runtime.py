"""Shared accelerator serialization for the local geometry worker.

Apple MPS is not safe for overlapping command encoders from the detector and
the differentiable pose solver in separate Python threads.  A single process
wide re-entrant lock keeps the GPU path deterministic while CPU request
admission and JPEG handling remain concurrent.
"""

from __future__ import annotations

import threading


GPU_COMPUTE_LOCK = threading.RLock()

