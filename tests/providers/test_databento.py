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

from liq.data.exceptions import ProviderNoDataError
from liq.data.providers.databento import (
    DATABENTO_BATCH_THRESHOLD_DAYS,
    DATABENTO_PRICE_SCALE,
    DATASET_ROUTING,
    AggregateCrossCheckError,
    DatabentoBarRecord,
    DatabentoError,
    DatabentoProvider,
    DatabentoRateLimitError,
    DatabentoSchemaError,
    DatabentoTransientError,
    _estimate_bytes_in,
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

    def test_fetch_bars_dataset_override_bypasses_routing(self) -> None:
        # Delisted names live only on venue datasets (e.g. XNAS.ITCH), not the
        # routed EQUS summary dataset; the override reaches them.
        provider, client = _make_provider()
        provider.fetch_bars(
            "DISCA",
            date(2020, 6, 1),
            date(2020, 6, 3),
            timeframe="1d",
            dataset="XNAS.ITCH",
        )
        assert client.timeseries.calls[0]["dataset"] == "XNAS.ITCH"

    def test_unsupported_timeframe_raises(self) -> None:
        from liq.data.exceptions import ProviderError

        provider, _ = _make_provider()
        with pytest.raises(ProviderError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="3m")


class TestNoDataTranslation:
    def test_422_no_data_error_maps_to_provider_no_data(self) -> None:
        class _NoDataError(Exception):
            http_status = 422
            json_body = {"error": "data_no_data_found_for_request"}

        provider, _ = _make_provider()

        with pytest.raises(ProviderNoDataError, match="No data"):
            provider._call_with_retry(
                lambda: (_ for _ in ()).throw(_NoDataError("No data was found")),
                sync_run_id="run-1",
                dataset="EQUS.MINI",
                symbol="FISV",
                request_kind="batch",
            )


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

    def test_bytes_in_estimate_skips_invalid_sdk_counters(self) -> None:
        class _Store:
            nbytes = "bad"
            num_bytes = -1
            bytes_in = 128

        assert _estimate_bytes_in(store=_Store(), rows=3) == 128
        assert _estimate_bytes_in(store=object(), rows=3) == 192


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

    def test_negative_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match="backoff_base_seconds"):
            DatabentoProvider(api_key="k", backoff_base_seconds=-0.1)


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


# ----- retry + transient error handling --------------------------------------


class _FlakyTimeseries:
    """``timeseries.get_range`` that raises a configured sequence of errors
    before yielding a normal response.

    Each entry in ``errors`` is raised once in order; once exhausted, the
    canned ``store`` is returned. ``calls`` exposes each attempt's args so
    tests can verify that retries actually re-invoke the API.
    """

    def __init__(self, store: _FakeDBNStore, errors: list[BaseException]) -> None:
        self._store = store
        self._errors = list(errors)
        self.calls: list[dict] = []

    def get_range(self, **kwargs) -> _FakeDBNStore:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return self._store


class _FlakyClient:
    def __init__(self, store: _FakeDBNStore, errors: list[BaseException]) -> None:
        self.timeseries = _FlakyTimeseries(store, errors)
        self.batch = _FakeBatch(store)


def _no_sleep(_seconds: float) -> None:
    """Drop-in for ``time.sleep`` so retry tests run instantly."""


class TestRetryOn5xxAndConnectionErrors:
    """5xx / connection-error retries with bounded backoff."""

    def test_succeeds_after_one_5xx(self) -> None:
        sleeps: list[float] = []
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[DatabentoTransientError("503")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=sleeps.append,
            max_retry_attempts=3,
        )
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert len(client.timeseries.calls) == 2  # 1 fail + 1 success
        assert sleeps  # at least one backoff occurred

    def test_exhausts_attempts_then_raises_last_error(self) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[DatabentoTransientError("503")] * 4)
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=_no_sleep,
            max_retry_attempts=3,
        )
        with pytest.raises(DatabentoTransientError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert len(client.timeseries.calls) == 3

    def test_does_not_retry_on_non_transient_error(self) -> None:
        from liq.data.exceptions import ProviderError

        class _PermanentError(ProviderError):
            pass

        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[_PermanentError("400 bad request")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=_no_sleep,
            max_retry_attempts=3,
        )
        with pytest.raises(_PermanentError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert len(client.timeseries.calls) == 1  # one attempt only


class TestRateLimitHandling:
    def test_429_with_retry_after_uses_provided_backoff(self) -> None:
        """A ``DatabentoRateLimitError`` with ``retry_after`` overrides the
        provider's exponential backoff so we honor the venue's hint."""
        sleeps: list[float] = []
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[DatabentoRateLimitError("429", retry_after=2.5)])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=sleeps.append,
            max_retry_attempts=3,
        )
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert sleeps == [2.5]

    def test_raw_429_response_is_translated_and_retried(self) -> None:
        """Production SDK exceptions carry HTTP response metadata rather
        than our custom exception types. A 429-shaped exception should
        still route through DatabentoRateLimitError retry handling."""

        class _Response:
            status_code = 429
            headers = {"Retry-After": "1.25"}

        class _HTTPError(Exception):
            response = _Response()

        sleeps: list[float] = []
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[_HTTPError("too many requests")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=sleeps.append,
            max_retry_attempts=3,
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        assert not df.is_empty()
        assert len(client.timeseries.calls) == 2
        assert sleeps == [1.25]

    def test_raw_5xx_response_is_translated_and_retried(self) -> None:
        class _Response:
            status_code = 503
            headers: dict[str, str] = {}

        class _HTTPError(Exception):
            response = _Response()

        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[_HTTPError("service unavailable")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=_no_sleep,
            max_retry_attempts=2,
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        assert not df.is_empty()
        assert len(client.timeseries.calls) == 2

    def test_transport_timeout_is_translated_and_retried(self) -> None:
        class ConnectTimeout(Exception):
            pass

        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[ConnectTimeout("timed out")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=_no_sleep,
            max_retry_attempts=2,
        )

        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

        assert not df.is_empty()
        assert len(client.timeseries.calls) == 2

    def test_databento_non_transient_error_is_not_rewrapped(self) -> None:
        provider, _ = _make_provider()
        err = DatabentoSchemaError("schema")
        assert provider._coerce_transient_error(err) is None


class TestSchemaMismatch:
    def test_schema_mismatch_raises_databento_schema_error(self) -> None:
        """If the response carries a schema that doesn't match the request
        (e.g. asked for ``ohlcv-1m`` but got ``ohlcv-1h``), the provider
        raises ``DatabentoSchemaError`` rather than silently mislabeling
        the bars."""
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        store.schema = "ohlcv-1h"  # type: ignore[attr-defined]
        client = _FakeClient(store)
        provider = DatabentoProvider(api_key="k", client=client)
        with pytest.raises(DatabentoSchemaError, match="ohlcv-1m"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

    def test_missing_schema_is_tolerated(self) -> None:
        """Fakes / older response shapes may not expose a ``schema``
        attribute. Missing → trust the caller's request (no mismatch
        flagged)."""
        provider, _ = _make_provider()
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()


class TestSyncRunIdCorrelation:
    def test_retry_event_shares_sync_run_id_with_fetch_event(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``databento_retry`` and the final ``databento_fetch`` carry the
        same ``sync_run_id`` so log readers can stitch the attempt sequence
        together."""
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _FlakyClient(store, errors=[DatabentoTransientError("503")])
        provider = DatabentoProvider(
            api_key="k",
            client=client,
            sleep_fn=_no_sleep,
            max_retry_attempts=3,
        )
        with caplog.at_level(logging.INFO, logger="liq.data.providers.databento"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        retries = [r for r in caplog.records if getattr(r, "event", None) == "databento_retry"]
        fetches = [r for r in caplog.records if getattr(r, "event", None) == "databento_fetch"]
        assert len(retries) == 1
        assert len(fetches) == 1
        assert retries[0].sync_run_id == fetches[0].sync_run_id  # type: ignore[attr-defined]


# ----- batch path: partial-recovery ------------------------------------------


class _StatefulBatch:
    """Track ``submit_job`` + ``get_job_details`` + ``download`` calls so
    partial-recovery tests can verify the right call sequence."""

    def __init__(
        self,
        store: _FakeDBNStore,
        *,
        download_error: BaseException | None = None,
        details_states: list[str] | None = None,
    ) -> None:
        self._store = store
        self._download_error = download_error
        self._details_states = list(details_states or ["done"])
        self.submit_calls: list[dict] = []
        self.detail_calls: list[str] = []
        self.download_calls: list[str] = []
        self.next_job: dict | None = None

    def submit_job(self, **kwargs) -> dict:
        self.submit_calls.append(kwargs)
        return self.next_job or {"id": f"job-{len(self.submit_calls)}", "state": "received"}

    def get_job_details(self, job_id: str) -> dict:
        self.detail_calls.append(job_id)
        state = self._details_states.pop(0) if self._details_states else "done"
        return {"id": job_id, "state": state}

    def download(self, *, job_id: str, **_kwargs) -> _FakeDBNStore:
        # Real SDK accepts ``output_dir`` and ``keep_zip`` kwargs; we
        # ignore them in the legacy store-returning fake. The provider
        # passes ``output_dir`` so the real path can land files on disk.
        self.download_calls.append(job_id)
        if self._download_error is not None:
            raise self._download_error
        return self._store


class _StatefulBatchClient:
    def __init__(self, store: _FakeDBNStore, **kwargs) -> None:
        self.timeseries = _FakeTimeseries(store)
        self.batch = _StatefulBatch(store, **kwargs)


def _batch_provider(
    tmp_path: pytest.TempPathFactory | object,
    client: _StatefulBatchClient,
) -> DatabentoProvider:
    """Build a provider rigged for batch-path tests:

    * ``batch_threshold_days=1`` → every span routes through batch.
    * ``batch_jobs_dir`` → ``tmp_path`` so the resume markers don't bleed
      into the real cache.
    """
    return DatabentoProvider(
        api_key="k",
        client=client,
        sleep_fn=_no_sleep,
        max_retry_attempts=1,
        batch_threshold_days=1,
        batch_jobs_dir=str(tmp_path),  # type: ignore[arg-type]
    )


class TestBatchPartialRecovery:
    def test_first_submit_persists_marker_and_returns_store(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _StatefulBatchClient(store)
        provider = _batch_provider(tmp_path, client)
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert len(client.batch.submit_calls) == 1
        # Marker is deleted on successful completion; no .json sidecars
        # left behind in the cache dir.
        leftovers = list(tmp_path.glob("*.json"))
        assert leftovers == []

    def test_download_failure_leaves_resumable_marker(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _StatefulBatchClient(
            store,
            download_error=RuntimeError("network blip mid-download"),
        )
        provider = _batch_provider(tmp_path, client)
        with pytest.raises(RuntimeError, match="network blip"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        # The job was submitted; the marker should be on disk for resume.
        leftovers = list(tmp_path.glob("*.json"))
        assert len(leftovers) == 1

    def test_resume_reuses_existing_job_no_resubmit(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())

        # First attempt: download fails, marker persists.
        flaky_client = _StatefulBatchClient(
            store,
            download_error=RuntimeError("transient"),
        )
        provider = _batch_provider(tmp_path, flaky_client)
        with pytest.raises(RuntimeError, match="transient"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        marker_files = list(tmp_path.glob("*.json"))
        assert len(marker_files) == 1
        original_marker = marker_files[0].read_text()

        # Second attempt with the same request signature: no resubmit,
        # just poll + download.
        recovered_client = _StatefulBatchClient(store)
        provider2 = _batch_provider(tmp_path, recovered_client)
        df = provider2.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert recovered_client.batch.submit_calls == []  # resumed, no new submit
        assert recovered_client.batch.detail_calls == ["job-1"]
        assert recovered_client.batch.download_calls == ["job-1"]
        # Marker cleaned up after success.
        assert list(tmp_path.glob("*.json")) == []
        # Sanity: the marker we examined did persist the right job id.
        assert "job-1" in original_marker

    def test_different_request_signature_does_not_resume(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())

        # Leave a marker behind for one symbol's request.
        flaky_client = _StatefulBatchClient(
            store,
            download_error=RuntimeError("oops"),
        )
        provider = _batch_provider(tmp_path, flaky_client)
        with pytest.raises(RuntimeError):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        # Different symbol → new submission, leave the AAPL marker alone.
        fresh_client = _StatefulBatchClient(store)
        provider2 = _batch_provider(tmp_path, fresh_client)
        df = provider2.fetch_bars("MSFT", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert len(fresh_client.batch.submit_calls) == 1
        # AAPL marker still present.
        leftover_names = {p.name for p in tmp_path.glob("*.json")}
        assert len(leftover_names) == 1

    def test_polling_waits_until_state_done(self, tmp_path) -> None:
        """When ``get_job_details`` reports non-terminal states, the
        provider polls (with backoff) until the job is ``done``."""
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _StatefulBatchClient(
            store,
            details_states=["queued", "processing", "done"],
        )
        provider = _batch_provider(tmp_path, client)
        df = provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert not df.is_empty()
        assert client.batch.detail_calls == ["job-1", "job-1", "job-1"]

    def test_missing_job_id_raises(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _StatefulBatchClient(store)
        client.batch.next_job = {"state": "received"}
        provider = _batch_provider(tmp_path, client)
        with pytest.raises(DatabentoError, match="job id"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")

    def test_terminal_failed_state_deletes_marker_and_raises(self, tmp_path) -> None:
        store = _FakeDBNStore(records=_sample_records(), symbology=_sample_symbology())
        client = _StatefulBatchClient(store, details_states=["failed"])
        provider = _batch_provider(tmp_path, client)
        with pytest.raises(DatabentoError, match="failed"):
            provider.fetch_bars("AAPL", date(2025, 1, 2), date(2025, 1, 3), timeframe="1m")
        assert list(tmp_path.glob("*.json")) == []


# ----- local 1m → nm aggregation helper -------------------------------------


def _minute_records(
    minutes: int,
    *,
    symbol: str = "AAPL",
    start_dt: datetime | None = None,
    open_seed: Decimal = Decimal("100.00"),
    step: Decimal = Decimal("0.01"),
    volume_seed: int = 100,
) -> list[DatabentoBarRecord]:
    """Build ``minutes`` consecutive 1-minute bars with deterministic OHLCV.

    Each bar's open/close walks by ``step``; high/low bracket the
    midpoint. Volume is ``volume_seed + i``.
    """
    start = start_dt or datetime(2025, 1, 2, 14, 30, tzinfo=UTC)
    out: list[DatabentoBarRecord] = []
    for i in range(minutes):
        ts_ns = int(start.timestamp() * 1e9) + i * 60 * 1_000_000_000
        open_ = open_seed + step * i
        close = open_seed + step * (i + 1)
        high = max(open_, close) + Decimal("0.05")
        low = min(open_, close) - Decimal("0.05")
        out.append(
            DatabentoBarRecord(
                ts_event_ns=ts_ns,
                instrument_id=12345,
                open_q9=int(open_ * DATABENTO_PRICE_SCALE),
                high_q9=int(high * DATABENTO_PRICE_SCALE),
                low_q9=int(low * DATABENTO_PRICE_SCALE),
                close_q9=int(close * DATABENTO_PRICE_SCALE),
                volume=volume_seed + i,
                symbol=symbol,
            )
        )
    return out


class TestAggregateBars:
    def test_5m_aggregation_bar_count_and_boundaries(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(_minute_records(60))
        df_5m = aggregate_bars(df_1m, timeframe="5m")
        assert df_5m.height == 12  # 60 minutes / 5 = 12
        # Every bar timestamp is aligned to a 5-minute boundary.
        ts_list = df_5m["timestamp"].to_list()
        assert all(ts.minute % 5 == 0 for ts in ts_list)

    def test_5m_ohlcv_integrity(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(_minute_records(5))
        df_5m = aggregate_bars(df_1m, timeframe="5m")
        assert df_5m.height == 1
        row = df_5m.row(0, named=True)
        assert row["open"] == Decimal("100.00000000")  # first.open
        assert row["close"] == Decimal("100.05000000")  # last.close
        # high = max(any.high), low = min(any.low)
        assert row["high"] == Decimal("100.10000000")  # last.high (close+0.05)
        assert row["low"] == Decimal("99.95000000")  # first.low (open-0.05)
        # volume = sum
        assert row["volume"] == Decimal("100") + Decimal("101") + Decimal("102") + Decimal(
            "103"
        ) + Decimal("104")

    def test_1h_aggregation_hour_aligned_input_collapses_cleanly(self) -> None:
        """When the 1m series starts on an hour boundary and spans an
        exact hour, ``1h`` aggregation produces one bar. (Mid-hour starts
        intentionally produce two partial bars — bucket alignment is
        ``group_by_dynamic`` semantics, not "first 60 from start".)"""
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(
            _minute_records(60, start_dt=datetime(2025, 1, 2, 14, 0, tzinfo=UTC))
        )
        df_1h = aggregate_bars(df_1m, timeframe="1h")
        assert df_1h.height == 1
        row = df_1h.row(0, named=True)
        assert row["volume"] == Decimal(sum(100 + i for i in range(60)))

    def test_1h_aggregation_mid_hour_start_produces_partial_buckets(self) -> None:
        """Documenting the contract: a 60-minute series starting at 14:30
        falls into the [14:00, 15:00) and [15:00, 16:00) buckets, yielding
        two partial bars. Downstream consumers can detect partials via
        bar count vs. window expectation."""
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(
            _minute_records(60, start_dt=datetime(2025, 1, 2, 14, 30, tzinfo=UTC))
        )
        df_1h = aggregate_bars(df_1m, timeframe="1h")
        assert df_1h.height == 2

    def test_1d_aggregation_collapses_full_session(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        # Two sessions, 5 minutes each, to verify day boundaries.
        day1 = _minute_records(
            5,
            start_dt=datetime(2025, 1, 2, 14, 30, tzinfo=UTC),
            volume_seed=10,
        )
        day2 = _minute_records(
            5,
            start_dt=datetime(2025, 1, 3, 14, 30, tzinfo=UTC),
            volume_seed=20,
        )
        df_1m = _records_to_dataframe(day1 + day2)
        df_1d = aggregate_bars(df_1m, timeframe="1d")
        assert df_1d.height == 2
        # First day's volume = sum(10..14) = 60.
        assert df_1d.row(0, named=True)["volume"] == Decimal("60")
        # Second day's volume = sum(20..24) = 110.
        assert df_1d.row(1, named=True)["volume"] == Decimal("110")

    def test_aggregate_unsupported_timeframe_raises(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(_minute_records(3))
        with pytest.raises(ValueError, match="timeframe"):
            aggregate_bars(df_1m, timeframe="7m")

    def test_aggregate_empty_frame_returns_empty_with_schema(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        df_empty = _records_to_dataframe([])
        out = aggregate_bars(df_empty, timeframe="5m")
        assert out.is_empty()
        assert set(out.columns) >= {"timestamp", "open", "high", "low", "close", "volume"}

    def test_aggregate_preserves_symbol_when_single_symbol(self) -> None:
        from liq.data.providers.databento import aggregate_bars

        df_1m = _records_to_dataframe(_minute_records(10))
        out = aggregate_bars(df_1m, timeframe="5m")
        assert out["symbol"].unique().to_list() == ["AAPL"]


# ----- validate_aggregate(symbol, date) cross-check --------------------------


class _DualSchemaTimeseries:
    """``get_range`` that dispatches by ``schema`` so a single fake client
    can return distinct stores for the 1m fetch + the 1d reference fetch."""

    def __init__(self, stores: dict[str, _FakeDBNStore]) -> None:
        self._stores = stores
        self.calls: list[dict] = []

    def get_range(self, *, schema: str, **kwargs) -> _FakeDBNStore:
        self.calls.append({"schema": schema, **kwargs})
        return self._stores[schema]


class _DualSchemaClient:
    def __init__(self, stores: dict[str, _FakeDBNStore]) -> None:
        self.timeseries = _DualSchemaTimeseries(stores)
        self.batch = _FakeBatch(next(iter(stores.values())))


def _daily_summary_record(
    *,
    symbol: str = "AAPL",
    open_: Decimal,
    high: Decimal,
    low: Decimal,
    close: Decimal,
    volume: int,
    ts_event_ns: int,
) -> DatabentoBarRecord:
    return DatabentoBarRecord(
        ts_event_ns=ts_event_ns,
        instrument_id=12345,
        open_q9=int(open_ * DATABENTO_PRICE_SCALE),
        high_q9=int(high * DATABENTO_PRICE_SCALE),
        low_q9=int(low * DATABENTO_PRICE_SCALE),
        close_q9=int(close * DATABENTO_PRICE_SCALE),
        volume=volume,
        symbol=symbol,
    )


class TestValidateAggregate:
    def test_fetch_time_validation_uses_existing_1m_frame(self) -> None:
        """``fetch_bars(..., validate_aggregate=True)`` performs the
        EQUS.SUMMARY cross-check without issuing a second 1m request."""
        records_1m = _minute_records(5, start_dt=datetime(2025, 1, 2, 14, 0, tzinfo=UTC))
        df_1m = _records_to_dataframe(records_1m)
        from liq.data.providers.databento import aggregate_bars

        daily_expected = aggregate_bars(df_1m, timeframe="1d").row(0, named=True)
        store_1m = _FakeDBNStore(records=records_1m, symbology=_sample_symbology())
        store_1d = _FakeDBNStore(
            records=[
                _daily_summary_record(
                    open_=daily_expected["open"],
                    high=daily_expected["high"],
                    low=daily_expected["low"],
                    close=daily_expected["close"],
                    volume=int(daily_expected["volume"]),
                    ts_event_ns=int(datetime(2025, 1, 2, tzinfo=UTC).timestamp() * 1e9),
                )
            ],
            symbology=_sample_symbology(),
        )
        client = _DualSchemaClient({"ohlcv-1m": store_1m, "ohlcv-1d": store_1d})

        provider = DatabentoProvider(api_key="k", client=client, sleep_fn=_no_sleep)
        df = provider.fetch_bars(
            "AAPL",
            date(2025, 1, 2),
            date(2025, 1, 2),
            timeframe="1m",
            validate_aggregate=True,
        )

        assert not df.is_empty()
        assert [call["schema"] for call in client.timeseries.calls] == ["ohlcv-1m", "ohlcv-1d"]

    def test_matching_1m_aggregate_passes(self) -> None:
        """When the locally-aggregated 1m bar matches EQUS.SUMMARY's 1d
        bar within the 0.001 % tolerance, ``validate_aggregate`` returns
        the comparison report without raising."""
        # 5-minute 1m series → expected daily OHLCV
        records_1m = _minute_records(5, start_dt=datetime(2025, 1, 2, 14, 0, tzinfo=UTC))
        df_1m = _records_to_dataframe(records_1m)
        from liq.data.providers.databento import aggregate_bars

        daily_expected = aggregate_bars(df_1m, timeframe="1d").row(0, named=True)

        # 1m store mirrors what real Databento would return for the day.
        store_1m = _FakeDBNStore(records=records_1m, symbology=_sample_symbology())
        # EQUS.SUMMARY 1d store matches exactly.
        summary_records = [
            _daily_summary_record(
                open_=daily_expected["open"],
                high=daily_expected["high"],
                low=daily_expected["low"],
                close=daily_expected["close"],
                volume=int(daily_expected["volume"]),
                ts_event_ns=int(datetime(2025, 1, 2, tzinfo=UTC).timestamp() * 1e9),
            )
        ]
        store_1d = _FakeDBNStore(records=summary_records, symbology=_sample_symbology())
        client = _DualSchemaClient({"ohlcv-1m": store_1m, "ohlcv-1d": store_1d})

        provider = DatabentoProvider(api_key="k", client=client, sleep_fn=_no_sleep)
        report = provider.validate_aggregate("AAPL", date(2025, 1, 2))
        assert report["matched"] is True
        assert report["symbol"] == "AAPL"

    def test_empty_local_aggregate_raises_data_quality_error(self) -> None:
        store_1m = _FakeDBNStore(records=[], symbology=_sample_symbology())
        store_1d = _FakeDBNStore(records=_sample_records(1), symbology=_sample_symbology())
        client = _DualSchemaClient({"ohlcv-1m": store_1m, "ohlcv-1d": store_1d})

        provider = DatabentoProvider(api_key="k", client=client, sleep_fn=_no_sleep)
        with pytest.raises(AggregateCrossCheckError, match="empty"):
            provider.validate_aggregate("AAPL", date(2025, 1, 2))

    def test_price_tolerance_uses_midrange_close_not_field_denominator(self) -> None:
        local_record = DatabentoBarRecord(
            ts_event_ns=int(datetime(2025, 1, 2, 14, 0, tzinfo=UTC).timestamp() * 1e9),
            instrument_id=12345,
            open_q9=int(Decimal("1.00000000") * DATABENTO_PRICE_SCALE),
            high_q9=int(Decimal("101.00000000") * DATABENTO_PRICE_SCALE),
            low_q9=int(Decimal("1.00000000") * DATABENTO_PRICE_SCALE),
            close_q9=int(Decimal("100.00000000") * DATABENTO_PRICE_SCALE),
            volume=100,
            symbol="AAPL",
        )
        summary_record = _daily_summary_record(
            open_=Decimal("1.00050000"),
            high=Decimal("101.00000000"),
            low=Decimal("1.00000000"),
            close=Decimal("100.00000000"),
            volume=100,
            ts_event_ns=int(datetime(2025, 1, 2, tzinfo=UTC).timestamp() * 1e9),
        )
        client = _DualSchemaClient(
            {
                "ohlcv-1m": _FakeDBNStore([local_record], _sample_symbology()),
                "ohlcv-1d": _FakeDBNStore([summary_record], _sample_symbology()),
            }
        )

        provider = DatabentoProvider(api_key="k", client=client, sleep_fn=_no_sleep)
        report = provider.validate_aggregate("AAPL", date(2025, 1, 2))

        assert report["matched"] is True
        assert Decimal(report["price_tolerance"]) == Decimal("0.0010000000000")

    def test_mismatch_above_tolerance_raises_data_quality_error(self) -> None:
        from liq.data.exceptions import DataError

        records_1m = _minute_records(5, start_dt=datetime(2025, 1, 2, 14, 0, tzinfo=UTC))
        store_1m = _FakeDBNStore(records=records_1m, symbology=_sample_symbology())

        # Perturb close by 1% — well above the 0.001 % tolerance.
        from liq.data.providers.databento import aggregate_bars

        daily = aggregate_bars(_records_to_dataframe(records_1m), timeframe="1d").row(0, named=True)
        summary_records = [
            _daily_summary_record(
                open_=daily["open"],
                high=daily["high"],
                low=daily["low"],
                close=daily["close"] * Decimal("1.01"),
                volume=int(daily["volume"]),
                ts_event_ns=int(datetime(2025, 1, 2, tzinfo=UTC).timestamp() * 1e9),
            )
        ]
        store_1d = _FakeDBNStore(records=summary_records, symbology=_sample_symbology())
        client = _DualSchemaClient({"ohlcv-1m": store_1m, "ohlcv-1d": store_1d})

        provider = DatabentoProvider(api_key="k", client=client, sleep_fn=_no_sleep)
        with pytest.raises(DataError, match="close") as exc_info:
            provider.validate_aggregate("AAPL", date(2025, 1, 2))
        assert isinstance(exc_info.value, AggregateCrossCheckError)
        assert "allowed" in exc_info.value.diffs["close"]


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
