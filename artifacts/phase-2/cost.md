# Databento SPY Pilot Cost

Date run: 2026-06-12

## Request

| Field | Value |
| --- | --- |
| Provider | Databento |
| Dataset | `EQUS.MINI` |
| Schema | `ohlcv-1m` |
| Symbol | `SPY` |
| Timeframe | `1m` |
| Provider start | `2024-06-03T00:00:00+00:00` |
| Provider end | `2024-06-04T23:59:59+00:00` |
| Route | `timeseries.get_range` |

## Result

| Metric | Value |
| --- | --- |
| Live smoke command | `RUN_DATABENTO=1 uv run pytest tests/providers/test_databento.py::test_real_databento_smoke -q --no-cov` |
| Live smoke result | Pass |
| Rows fetched | `1100` |
| Record count estimate | `1100` |
| Billable size estimate | `61600` bytes |
| Databento cost estimate | `$0.000688` |
| First timestamp | `2024-06-03 11:00:00+00:00` |
| Last timestamp | `2024-06-04 23:58:00+00:00` |

## Notes

- The default single-test pytest invocation without `--no-cov` executed
  the live smoke successfully but failed the process-level coverage
  threshold because only one test was selected. The command above is the
  result-bearing live smoke command.
- The cost was queried through Databento metadata for the exact
  timestamp bounds used by `DatabentoProvider.fetch_bars(...)`, which
  treats the `end` date as inclusive through `23:59:59`.
- The API key was loaded from `.env` and was not printed.
