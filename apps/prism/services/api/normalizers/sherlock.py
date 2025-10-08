"""Normalization helpers for the sherlock collector."""
from __future__ import annotations

from ..models import CollectorResult


def normalize(result: CollectorResult) -> CollectorResult:
    """Return the collector result untouched.

    The collectors already emit normalized :class:`CollectorResult` objects.
    This function exists to preserve the extension point expected by
    downstream modules and future paid integrations.
    """

    return result
