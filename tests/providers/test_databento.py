"""TDD tests for ``DatabentoProvider``.

Locked contract:

* Dataset routing literal ``{(equity, 1m): EQUS.MINI, (equity, 1d): EQUS.SUMMARY}``.
* Batch-vs-sync routing by date span (``DATABENTO_BATCH_THRESHOLD_DAYS = 14``).
* Integer → Decimal normalization with no float round-trip.
* Symbology persistence to ``liq-store`` under ``reference/databento/symbology``.
* One structured log event ``databento_fetch`` per fetch.

Real Databento API is NEVER called in the default test run; every test
here uses an in-process fake client so the suite is deterministic and
offline. The only test that talks to the real venue is gated behind
``@pytest.mark.databento`` (env var ``RUN_DATABENTO=1``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from liq.data.providers.databento import (
    DATABENTO_BATCH_THRESHOLD_DAYS,
    DATABENTO_PRICE_SCALE,
    DATASET_ROUTING,
    DatabentoBarRecord,
    DatabentoProvider,
    _records_to_dataframe,
)

# ----- fake databento client -------------------------------------------------


@dataclass
class _FakeDBNStore:
    """Mimic the bits of ``databento.DBNStore`` the provider consumes.

    The real ``DBNStore`` supports ``for msg in store`` and exposes
    ``symbology`` as a ``{raw_symbol: [{instrument_id, valid_from,
    valid_to}, ...]}`` mapping. The fake mirrors both.
    """

    records: list[DatabentoBarRecord]
    symbology: dict[str, list[dict]]

    def __iter__(self):
        return iter(self.records)

    def replay(self) -> list[DatabentoBarRecord]:
        # Retained for older tests that bind to the legacy replay name;
        # the provider preferentially uses ``__iter__``.
        return list(self.records)


class _FakeTimeseries:
    def __init__(self, store: _FakeDBNStore) -> None:
        self._store = store
        self.calls: list[dict] = []

    def get_range(
        self,
        *,
        dataset: str,
        symbols: list[str] | str,
        schema: str,
        start: datetime,
        end: datetime,
    ) -> _FakeDBNStore:
        self.calls.append(
            {
                "dataset": dataset,
                "symbols": symbols,
                "schema": schema,
                "start": start,
                "end": end,
            }
        )
        return self._store


class _FakeBatch:
    def __init__(self, store: _FakeDBNStore) -> None:
        self._store = store
        self.submit_calls: list[dict] = []

    def submit_job(self, **kwargs) -> dict:
        self.submit_calls.append(kwargs)
        return {"id": "job-1", "state": "received"}

    def get_job_details(self, job_id: str) -> dict:
        return {"id": job_id, "state": "done"}

    def download(self, **kwargs) -> _FakeDBNStore:
        return self._store


class _FakeClient:
    """Drop-in for ``databento.Historical`` in unit tests."""

    def __init__(self, store: _FakeDBNStore) -> None:
        self.timeseries = _FakeTimeseries(store)
        self.batch = _FakeBatch(store)


# ----- helpers ---------------------------------------------------------------


def _sample_records(count: int = 2, symbol: str = "AAPL") -> list[DatabentoBarRecord]:
    """Build representative OHLCV-1m fake records.

    Prices are integer-scaled by 1e9 per the Databento DBN convention; the
    provider must convert to Decimal exactly (no float intermediates).
    """
    base_ns = int(datetime(2025, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1e9)
    out: list[DatabentoBarRecord] = []
    for i in range(count):
        out.append(
            DatabentoBarRecord(
                ts_event_ns=base_ns + i * 60 * 1_000_000_000,  # +1 min
                instrument_id=12345,
                open_q9=int(Decimal("150.25") * DATABENTO_PRICE_SCALE),
                high_q9=int(Decimal("150.75") * DATABENTO_PRICE_SCALE),
                low_q9=int(Decimal("150.10") * DATABENTO_PRICE_SCALE),
                close_q9=int(Decimal("150.50") * DATABENTO_PRICE_SCALE),
                volume=1_000 + i,
                symbol=symbol,
            )
        )
    return out


def _sample_symbology(symbol: str = "AAPL") -> dict[str, list[dict]]:
    return {
        symbol: [
            {
                "instrument_id": 12345,
                "valid_from": "2024-01-01",
                "valid_to": "2099-12-31",
            }
        ],
    }


def _make_provider(
    records: list[DatabentoBarRecord] | None = None,
    symbology: dict | None = None,
) -> tuple[DatabentoProvider, _FakeClient]:
    # ``is None`` (not ``or``) so ``records=[]`` / ``symbology={}`` don't
    # silently fall back to defaults — the empty cases are explicitly
    # tested elsewhere.
    store = _FakeDBNStore(
        records=_sample_records() if records is None else records,
        symbology=_sample_symbology() if symbology is None else symbology,
    )
    client = _FakeClient(store)
    provider = DatabentoProvider(api_key="test-key", client=client)
    return provider, client


# ----- dataset routing -------------------------------------------------------


class TestDatasetRouting:
    def test_equity_1m_maps_to_equs_mini(self) -> None:
        assert DATASET_ROUTING[("equity", "1m")] == "EQUS.MINI"

    def test_equity_1d_maps_to_equs_summary(self) -> None:
        assert DATASET_ROUTING[("equity", "1d")] == "EQUS.SUMMARY"

    def test_default_asset_class_is_equity(self) -> None:
        provider, _ = _make_provider()
        assert provider.asset_class == "equity"

    def test_fetch_bars_uses_routed_dataset(self) -> None:
        provider, client = _make_provider()
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert client.timeseries.calls[0]["dataset"] == "EQUS.MINI"

    def test_unsupported_timeframe_raises(self) -> None:
        from liq.data.exceptions import ProviderError

        provider, _ = _make_provider()
        with pytest.raises(ProviderError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="3m")


# ----- batch-vs-sync routing -------------------------------------------------


class TestBatchThresholdRouting:
    def test_default_threshold_is_14_days(self) -> None:
        assert DATABENTO_BATCH_THRESHOLD_DAYS == 14

    def test_under_threshold_uses_get_range(self) -> None:
        provider, client = _make_provider()
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert len(client.timeseries.calls) == 1
        assert client.batch.submit_calls == []

    def test_at_or_above_threshold_uses_batch(self) -> None:
        provider, client = _make_provider()
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 18), timeframe="1m")
        assert client.batch.submit_calls != []
        assert client.timeseries.calls == []

    def test_threshold_is_configurable(self) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FakeClient(store)
        provider = DatabentoProvider(api_key="k", client=client, batch_threshold_days=3)
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 8), timeframe="1m")
        assert client.batch.submit_calls != []


# ----- normalization (integer → Decimal exact) -------------------------------


class TestNormalization:
    def test_prices_are_decimal_no_float_intermediate(self) -> None:
        provider, _ = _make_provider()
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert df.schema["open"] == pl.Decimal(38, 8)
        assert df.schema["high"] == pl.Decimal(38, 8)
        assert df.schema["low"] == pl.Decimal(38, 8)
        assert df.schema["close"] == pl.Decimal(38, 8)

    def test_known_records_round_trip_exact_values(self) -> None:
        provider, _ = _make_provider()
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        row = df.row(0, named=True)
        assert row["symbol"] == "AAPL"
        assert row["open"] == Decimal("150.25")
        assert row["high"] == Decimal("150.75")
        assert row["low"] == Decimal("150.10")
        assert row["close"] == Decimal("150.50")
        assert row["volume"] == Decimal("1000")

    def test_timestamps_are_utc_aware(self) -> None:
        provider, _ = _make_provider()
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        ts_dtype = df["timestamp"].dtype
        assert isinstance(ts_dtype, pl.Datetime)
        assert ts_dtype.time_zone == "UTC"

    def test_empty_records_yield_empty_dataframe_with_schema(self) -> None:
        provider, _ = _make_provider(records=[])
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert df.is_empty()
        assert set(df.columns) >= {"timestamp", "open", "high", "low", "close", "volume", "symbol"}

    def test_records_to_dataframe_handles_mixed_symbols(self) -> None:
        # The batch path can return multiple symbols in one DBN store;
        # the converter must keep the symbol column intact (downstream
        # cross-sectional reads rely on it; FR-7 of the requirements).
        records = _sample_records(symbol="AAPL") + _sample_records(symbol="MSFT")
        df = _records_to_dataframe(records)
        assert set(df["symbol"].to_list()) == {"AAPL", "MSFT"}


# ----- symbology persistence -------------------------------------------------


class _FakeStore:
    """Capture writes so symbology persistence can be asserted."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, pl.DataFrame, str]] = []

    def write(self, key: str, data: pl.DataFrame, mode: str = "append") -> None:
        self.writes.append((key, data, mode))


class TestSymbologyPersistence:
    def test_symbology_written_under_reference_key(self) -> None:
        store = _FakeStore()
        provider, _ = _make_provider()
        provider._store = store  # type: ignore[assignment]
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        keys = [w[0] for w in store.writes]
        assert "reference/databento/symbology" in keys

    def test_symbology_payload_carries_required_columns(self) -> None:
        store = _FakeStore()
        provider, _ = _make_provider()
        provider._store = store  # type: ignore[assignment]
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        sym_writes = [w for w in store.writes if w[0] == "reference/databento/symbology"]
        assert sym_writes, "expected at least one symbology write"
        sym_df = sym_writes[0][1]
        assert set(sym_df.columns) >= {
            "raw_symbol",
            "instrument_id",
            "valid_from",
            "valid_to",
        }
        row = sym_df.row(0, named=True)
        assert row["raw_symbol"] == "AAPL"
        assert row["instrument_id"] == 12345

    def test_no_symbology_write_when_empty(self) -> None:
        store = _FakeStore()
        provider, _ = _make_provider(symbology={})
        provider._store = store  # type: ignore[assignment]
        provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not any(w[0] == "reference/databento/symbology" for w in store.writes)


# ----- logging --------------------------------------------------------------


class TestLogging:
    def test_emits_databento_fetch_event(self, caplog: pytest.LogCaptureFixture) -> None:
        provider, _ = _make_provider()
        with caplog.at_level(logging.INFO, logger="liq.data.providers.databento"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        events = [r for r in caplog.records if getattr(r, "event", None) == "databento_fetch"]
        assert events, "expected one INFO log with event=databento_fetch"
        rec = events[0]
        for field in (
            "provider",
            "dataset",
            "symbols_count",
            "start",
            "end",
            "request_kind",
            "bytes_in",
            "duration_ms",
            "sync_run_id",
        ):
            assert hasattr(rec, field), f"missing field {field!r} on databento_fetch event"
        assert rec.bytes_in >= 0
        assert rec.sync_run_id

    def test_request_kind_distinguishes_get_range_vs_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider, _ = _make_provider()
        with caplog.at_level(logging.INFO, logger="liq.data.providers.databento"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 18), timeframe="1m")
        events = [r for r in caplog.records if getattr(r, "event", None) == "databento_fetch"]
        kinds = {e.request_kind for e in events}
        assert kinds == {"get_range", "batch"}


# ----- hypothesis: integer ↔ Decimal exactness -------------------------------


class TestHypothesisIntegerDecimalRoundTrip:
    @settings(max_examples=50, deadline=None)
    @given(
        open_q9=st.integers(min_value=0, max_value=10**18),
        high_q9=st.integers(min_value=0, max_value=10**18),
        low_q9=st.integers(min_value=0, max_value=10**18),
        close_q9=st.integers(min_value=0, max_value=10**18),
        volume=st.integers(min_value=0, max_value=10**12),
    )
    def test_integer_inputs_round_trip_to_decimal_without_float_drift(
        self,
        open_q9: int,
        high_q9: int,
        low_q9: int,
        close_q9: int,
        volume: int,
    ) -> None:
        rec = DatabentoBarRecord(
            ts_event_ns=int(datetime(2025, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1e9),
            instrument_id=1,
            open_q9=open_q9,
            high_q9=high_q9,
            low_q9=low_q9,
            close_q9=close_q9,
            volume=volume,
            symbol="X",
        )
        df = _records_to_dataframe([rec])
        # Exact reconstruction at the schema-declared scale of 8 places.
        # The provider quantizes the (Decimal int / 1e9) result to match
        # PRICE_DTYPE; we mirror that quantization here so equality holds.
        quant = Decimal("1e-8")
        expected_open = (Decimal(open_q9) / DATABENTO_PRICE_SCALE).quantize(quant)
        assert df.row(0, named=True)["open"] == expected_open


# ----- error paths ----------------------------------------------------------


class TestErrorPaths:
    def test_missing_api_key_raises_at_factory(self) -> None:
        from liq.data.settings import LiqDataSettings, create_databento_provider

        s = LiqDataSettings(databento_api_key=None)
        with pytest.raises(ValueError, match="DATABENTO_API_KEY"):
            create_databento_provider(s)

    def test_empty_symbol_raises(self) -> None:
        provider, _ = _make_provider()
        with pytest.raises(ValueError):
            provider.fetch_bars("", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

    def test_end_before_start_raises(self) -> None:
        provider, _ = _make_provider()
        with pytest.raises(ValueError):
            provider.fetch_bars("AAPL", date(2025, 1, 3), date(2025, 1, 2), timeframe="1m")


# ----- provider metadata ----------------------------------------------------


class TestProviderMetadata:
    def test_name(self) -> None:
        provider, _ = _make_provider()
        assert provider.name == "databento"

    def test_supported_asset_classes(self) -> None:
        provider, _ = _make_provider()
        assert "equity" in provider.supported_asset_classes

    def test_supported_timeframes_include_1m_and_1d(self) -> None:
        provider, _ = _make_provider()
        assert {"1m", "1d"}.issubset(set(provider.supported_timeframes))

    def test_list_instruments_returns_empty_schema_frame(self) -> None:
        """``list_instruments`` is a protocol stub here; returns an empty frame
        rather than ``NotImplementedError`` so the protocol stays consumable."""
        provider, _ = _make_provider()
        df = provider.list_instruments()
        assert df.is_empty()
        assert {"symbol", "asset_class", "exchange"}.issubset(set(df.columns))

    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            DatabentoProvider(api_key="")


# ----- shape-tolerance edge cases (symbology + record extraction) ------------


class TestSymbologyShapeTolerance:
    def test_non_dict_symbology_returns_empty_inverse(self) -> None:
        """When the wire returned something unexpected (string, list, None),
        the inverse-lookup falls back to empty rather than raising."""
        # ``_build_instrument_to_symbol`` is the load-bearing helper; bypass
        # via a custom store object.
        provider, _ = _make_provider()

        class _BadStore:
            symbology = "not-a-dict"
            records: list = []

            def __iter__(self):
                return iter(())

        inverse = provider._build_instrument_to_symbol(_BadStore())
        assert inverse == {}

    def test_symbology_value_not_a_list_is_skipped(self) -> None:
        """Defensive: an entry whose value is neither list nor dict gets skipped."""
        from liq.data.providers.databento import _symbology_to_dataframe

        df = _symbology_to_dataframe({"AAPL": "garbage"})
        assert df.is_empty()

    def test_symbology_entry_not_a_dict_is_skipped(self) -> None:
        from liq.data.providers.databento import _symbology_to_dataframe

        df = _symbology_to_dataframe({"AAPL": ["not-a-dict"]})
        assert df.is_empty()

    def test_symbology_entry_missing_instrument_id_is_skipped(self) -> None:
        from liq.data.providers.databento import _symbology_to_dataframe

        df = _symbology_to_dataframe({"AAPL": [{"valid_from": "2024-01-01"}]})
        assert df.is_empty()

    def test_symbology_instrument_id_not_intable_is_skipped(self) -> None:
        from liq.data.providers.databento import _symbology_to_dataframe

        df = _symbology_to_dataframe({"AAPL": [{"instrument_id": "abc"}]})
        assert df.is_empty()


class TestExtractRecordsLegacyReplay:
    """Compatibility hook: stores that pre-date the iterator contract."""

    def test_replay_fallback_when_store_not_iterable(self) -> None:
        @dataclass
        class _ReplayOnlyStore:
            records: list[DatabentoBarRecord]
            symbology: dict

            def replay(self) -> list[DatabentoBarRecord]:
                return list(self.records)

        store = _ReplayOnlyStore(
            records=_sample_records(),
            symbology=_sample_symbology(),
        )
        client = _FakeClient(_FakeDBNStore(records=[], symbology={}))
        provider = DatabentoProvider(api_key="k", client=client)
        # Hand the legacy-shape store straight to the extractor.
        out = provider._extract_records(store, fallback_symbol="AAPL")
        assert len(out) == 2
        assert all(r.symbol == "AAPL" for r in out)

    def test_no_replay_no_iter_yields_empty(self) -> None:
        provider, _ = _make_provider()

        class _Inert:
            symbology: dict = {}

        out = provider._extract_records(_Inert(), fallback_symbol="X")
        assert out == []


# ----- real-API smoke (opt-in) ----------------------------------------------


@pytest.mark.databento
@pytest.mark.skipif(
    os.environ.get("RUN_DATABENTO") != "1",
    reason="set RUN_DATABENTO=1 (and DATABENTO_API_KEY) to run the real-API smoke",
)
def test_real_databento_smoke() -> None:
    """Optional smoke against the live Databento API.

    Default test runs MUST NOT call Databento; this test is gated on
    ``RUN_DATABENTO=1`` so it stays opt-in. The smoke verifies the
    provider's wiring against a real, deliberately tiny window — one
    trading day of SPY 1m bars (~390 records, a few KB on the wire).
    Single symbol, under the batch threshold, so it routes through
    ``timeseries.get_range``. SPY is the workspace's canonical
    smoke-test symbol; keep this single fetch — broader real-API
    sweeps would burn credits.
    """
    from liq.data.settings import create_databento_provider

    provider = create_databento_provider()
    end = date(2024, 6, 4)
    start = end - timedelta(days=1)
    df = provider.fetch_bars("SPY", start, end, timeframe="1m")
    assert df.height > 0
    assert df["symbol"].unique().to_list() == ["SPY"]
