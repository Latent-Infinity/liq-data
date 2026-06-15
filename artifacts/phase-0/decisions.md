# Phase 0 — liq-data decisions

## Decisions

- **`classify_gap` tie-breaker for single mid-week holidays.** A gap whose
  only interior day is a weekday holiday is labeled `pre_holiday` (the
  c_prev session was the last trading day before the holiday). This is
  the more economically meaningful framing and matches the cardinal
  example used in the research plan (Wed July 3 close → Fri July 5
  open). `post_holiday` is reserved for the rare case where the holiday
  is adjacent to `o` but not to `c_prev` (research plan §3.1a does not
  disambiguate; we chose the side that is testable and dominant in
  practice).
- **`closed_hours_between` returns raw UTC subtraction.** No calendar
  walk over the closed interval — callers pass session-boundary
  timestamps and we trust them. This keeps the helper trivial and
  composable with `nyse_session_close` / `session_open`. DST is absorbed
  naturally because UTC subtraction is timezone-naive.
- **Same-session gap is `halt_reopen` regardless of magnitude.** Even a
  multi-hour halt within one trading session is `halt_reopen`, not
  `weeknight`. The session-date equality is the discriminator.
- **Mixed weekend + holiday → `long_holiday`.** Good Friday weekend,
  MLK weekend, Memorial Day weekend, Thanksgiving Wed→Mon all fold
  into `long_holiday`. This matches the research plan's ~89h cardinal
  example.

## Deviations

None.
