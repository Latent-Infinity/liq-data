"""Symbology rename round-trip tests.

A raw symbol can map to different instrument ids across a backfill
window (e.g., ``FB → META`` in 2022). The provider's
``_persist_symbology`` writes one row per mapping with
``valid_from / valid_to`` columns; this module pins the read-side
contract:

* Persisted rows survive the full date span and can be reloaded.
* ``DatabentoProvider.resolve_symbol_for_date(symbol, as_of)`` returns
  the instrument id whose validity window contains ``as_of``.
* When no row covers ``as_of``, the helper returns ``None`` so callers
  fail loud rather than silently picking the wrong id.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from liq.data.providers.databento import (
    SYMBOLOGY_KEY,
    DatabentoProvider,
)
from liq.store.parquet import ParquetStore


@pytest.fixture
def store(tmp_path: Path) -> ParquetStore:
    return ParquetStore(str(tmp_path))


class _RenameStore:
    """In-memory DBNStore-shape with multi-row symbology for one raw symbol."""

    def __init__(self) -> None:
        # ``FB`` was renamed to ``META`` mid-year 2022 (instrument id
        # changed at the venue).
        self.symbology = {
            "mappings": {
                "FB": [
                    {
                        "start_date": "2022-01-01",
                        "end_date": "2022-06-08",
                        "symbol": "1001",
                    },
                ],
                "META": [
                    {
                        "start_date": "2022-06-09",
                        "end_date": "2099-12-31",
                        "symbol": "2002",
                    },
                ],
            }
        }

    def __iter__(self):
        return iter(())


class TestSymbologyRenamePersistence:
    def test_persist_round_trips_multi_row_mappings(self, store: ParquetStore) -> None:
        provider = DatabentoProvider(api_key="x", client=None, store=store)
        provider._persist_symbology(_RenameStore())

        assert store.exists(SYMBOLOGY_KEY)
        df = store.read(SYMBOLOGY_KEY)
        assert df.height == 2
        assert sorted(df["raw_symbol"].to_list()) == ["FB", "META"]
        assert set(df["instrument_id"].to_list()) == {1001, 2002}


class TestResolveSymbolForDate:
    def _setup_provider(self, store: ParquetStore) -> DatabentoProvider:
        provider = DatabentoProvider(api_key="x", client=None, store=store)
        provider._persist_symbology(_RenameStore())
        return provider

    def test_returns_pre_rename_id_for_pre_rename_date(self, store: ParquetStore) -> None:
        provider = self._setup_provider(store)
        assert provider.resolve_symbol_for_date("FB", date(2022, 3, 15)) == 1001

    def test_returns_post_rename_id_for_post_rename_date(self, store: ParquetStore) -> None:
        provider = self._setup_provider(store)
        assert provider.resolve_symbol_for_date("META", date(2023, 1, 4)) == 2002

    def test_returns_none_for_unknown_symbol(self, store: ParquetStore) -> None:
        provider = self._setup_provider(store)
        assert provider.resolve_symbol_for_date("NOPE", date(2023, 1, 4)) is None

    def test_returns_none_when_date_outside_any_window(self, store: ParquetStore) -> None:
        provider = self._setup_provider(store)
        # 2020 is before both validity windows.
        assert provider.resolve_symbol_for_date("FB", date(2020, 1, 1)) is None

    def test_empty_valid_to_is_open_ended(self, store: ParquetStore) -> None:
        provider = DatabentoProvider(api_key="x", client=None, store=store)
        store.write(
            SYMBOLOGY_KEY,
            pl.DataFrame(
                [
                    {
                        "raw_symbol": "META",
                        "instrument_id": 2002,
                        "valid_from": "2022-06-09",
                        "valid_to": "",
                    }
                ]
            ),
            mode="append",
        )

        assert provider.resolve_symbol_for_date("META", date(2024, 1, 4)) == 2002

    def test_returns_none_when_store_has_no_symbology(self, store: ParquetStore) -> None:
        provider = DatabentoProvider(api_key="x", client=None, store=store)
        assert provider.resolve_symbol_for_date("FB", date(2022, 3, 15)) is None
