"""Outcome-free cohort candle metadata; this module grants no source-use authority."""

from dataclasses import dataclass

from pydantic import ValidationError

from liq.data.fx_fix_cohort_window import CohortSourceWindow
from liq.data.oanda_source_envelope import CaptureSecrets, _parse_source_envelope
from liq.data.oanda_source_quality import (
    NativeCandleQuality,
    SourceQualityRefused,
    _project_source_candles,
)


@dataclass(frozen=True, slots=True)
class NativeCohortSourceQuality:
    """Native source-only projection, distinct from the legacy six-date QA type."""

    window: CohortSourceWindow
    candles: tuple[NativeCandleQuality, ...]


def _reparse_cohort_window(window: CohortSourceWindow) -> CohortSourceWindow:
    try:
        return CohortSourceWindow.model_validate(window.model_dump(warnings=False))
    except ValidationError:
        raise SourceQualityRefused("WINDOW_REFUSED") from None


def inspect_cohort_source_quality(
    raw: bytes, window: CohortSourceWindow, secrets: CaptureSecrets
) -> NativeCohortSourceQuality:
    """Reparse fixed cohort scope, then inspect bounded secret-safe native metadata."""
    validated = _reparse_cohort_window(window)
    envelope = _parse_source_envelope(
        raw, validated.symbol, (secrets.account_id, secrets.api_token)
    )
    return NativeCohortSourceQuality(
        window=validated, candles=_project_source_candles(envelope, validated.query())
    )


def inspect_quarantined_cohort_source_quality(
    raw: bytes, window: CohortSourceWindow
) -> NativeCohortSourceQuality:
    """Inspect only independently authorized, original-witness-verified quarantine.

    The caller must hold and revalidate original private source, hash, full file
    state, audit and protocol witnesses before and after use. This is not a secret
    scan, retention permission, fixture promotion or substitute for those checks.
    """
    validated = _reparse_cohort_window(window)
    envelope = _parse_source_envelope(raw, validated.symbol, None)
    return NativeCohortSourceQuality(
        window=validated, candles=_project_source_candles(envelope, validated.query())
    )
