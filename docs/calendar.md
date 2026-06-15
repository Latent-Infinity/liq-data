# Calendar helpers (`liq.data.calendar`)

UTC-only, NYSE/XNYS-aware wrappers around `exchange_calendars`. Consumers
treat session boundaries, gaps, and DST through this module rather than
importing `exchange_calendars` directly.

## Trailing-window helpers

| Helper | Returns | Notes |
| --- | --- | --- |
| `trading_minutes_window(end, n)` | `(start, end)` half-open window over the previous `n` trading minutes | Skips weekends/holidays/early closes; UTC inputs/outputs. |
| `extended_trading_minutes_window(end, n)` | Same shape, `04:00`–`20:00` ET grid | DST-aware. |
| `trading_sessions_window(end, n)` | `(start, end)` over previous `n` sessions | `start` is the open of the earliest session. |
| `nyse_session_close(session)` | UTC close for a `date` session | Honors early closes. |

## Closed-market gap helpers

These wrappers expose the trading calendar's view of the closed interval
between a previous session close (`c_prev`) and the next session open
(`o`). They are consumed by `liq-features` to annotate
`overnight_gap_var_t` with its closed-market context — the raw empirical
gap feeds the canonical risk variance unscaled, while these helpers
provide the diagnostic / classification used by regime logic.

### `closed_hours_between(c_prev, o) -> float`

Returns the elapsed hours `(o - c_prev)` over the closed interval.
Calendar-derived in the sense that callers typically obtain `c_prev` /
`o` from session boundaries on this module; the arithmetic itself is a
UTC subtraction that absorbs DST naturally.

Worked examples (UTC):

| Scenario | `c_prev` | `o` | Result |
| --- | --- | --- | --- |
| Weeknight (standard time) | `2025-01-15 21:00` | `2025-01-16 14:30` | `17.5` h |
| Weeknight (DST) | `2024-06-05 20:00` | `2024-06-06 13:30` | `17.5` h |
| Weekend (no holiday) | `2024-06-07 20:00` | `2024-06-10 13:30` | `65.5` h |
| Weekend spanning DST spring-forward | `2024-03-08 21:00` | `2024-03-11 13:30` | `64.5` h |
| 3-day weekend (MLK Mon) | `2025-01-17 21:00` | `2025-01-21 14:30` | `89.5` h |
| Intraday halt + reopen | `2024-06-03 17:00` | `2024-06-03 18:30` | `1.5` h |

Both timestamps must be timezone-aware; `o` must be at or after `c_prev`.

### `classify_gap(c_prev, o) -> GapClass`

Returns one of the `GapClass` literals — `weeknight`, `weekend`,
`pre_holiday`, `post_holiday`, `long_holiday`, `halt_reopen` — using the
XNYS calendar to inspect what kinds of non-trading days fall inside the
gap.

| Label | Cardinal example |
| --- | --- |
| `halt_reopen` | `c_prev` and `o` share a trading session date. |
| `weeknight` | Wed close → Thu open (adjacent sessions, no calendar gap). |
| `weekend` | Fri close → Mon open with only Sat/Sun interior, no holiday. |
| `pre_holiday` | Wed `2024-07-03` close → Fri `2024-07-05` open (Thu July 4 is a single mid-week holiday adjacent to `c_prev`). |
| `post_holiday` | Reserved for the rare case where the holiday is adjacent only to `o` (no weekend interior). |
| `long_holiday` | Fri close → Tue open over a 3-day weekend (MLK / Good Friday / Memorial Day), or any gap mixing weekend + holiday days. |

The classification feeds `gap_class_t` in the volatility decomposition;
it never scales the canonical `risk_var_t` — duration nuance is carried
through `gap_var_per_closed_hour_t = overnight_gap_var_t / (closed_hours_t + eps)`
as a diagnostic only.
