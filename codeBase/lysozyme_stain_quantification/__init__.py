"""Compatibility shim for legacy imports.

Maps `lysozyme_stain_quantification.*` imports to the canonical
`codeBase/crypt_detection_code/lysozyme_stain_quantification` package.
"""

from __future__ import annotations

from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "crypt_detection_code" / "lysozyme_stain_quantification"
__path__ = [str(_TARGET)]
