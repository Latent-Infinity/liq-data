# TradeStation symbol collision: equity `CL` / `ES` versus continuous futures

**Observed:** 2026-08-12
**Status:** QUARANTINED — affected keys are not admissible as equity evidence.
**Detection context:** non-signal refresh of existing TradeStation 1-minute keys.

## Finding

`TradeStationProvider._normalize_symbol` unconditionally maps the bare symbols
`CL` and `ES` to the continuous-futures API symbols `@CL` and `@ES`. The same
bare symbols identify Clorox (`CL`) and Eversource Energy (`ES`) in the stored
US-equity universe. The storage key does not carry asset class, so a request for
either equity is written beneath the same key that a continuous-futures request
would use.

The collision applies to every TradeStation timeframe, not only 1-minute bars.
At detection time, ambiguous `CL` and `ES` keys existed under both configured
data roots; the `financial_data` root contained both `1m` and `1d` keys. The
2026-08-12 rolling-window refresh also routed those two financial-data requests
through `@CL` and `@ES` before the collision was detected.

No price values or returns were inspected during detection. The evidence was
the provider's deterministic normalization rule, the equity-universe symbols,
and storage filenames/keys.

## Immediate disposition

- Treat every bare TradeStation `CL` / `ES` key as ambiguous and exclude it from
  equity research, validation, coverage claims, and fixtures.
- Do not rename, delete, splice, or reinterpret existing files silently.
- Do not substitute another provider or symbol without a separately reviewed
  data-source decision.
- Preserve the 2026-08-12 refresh logs as acquisition provenance.

## Required remediation

1. Make TradeStation symbol normalization asset-class aware. Bare equity symbols
   must remain `CL` / `ES`; continuous futures must require an explicit futures
   identity or explicit `@` API symbol.
2. Make storage identity asset-class aware, or otherwise prevent equities and
   continuous futures from sharing a key.
3. Add regression tests for both sides of each collision.
4. Re-fetch the equity keys from an explicitly equity-routed request, validate
   them independently, and only then replace/quarantine the ambiguous files by
   an operator-approved migration.
5. Audit every stored TradeStation key against the provider's continuous-futures
   shorthand list (`ES`, `NQ`, `CL`, `GC`, `SI`, `ZB`, `ZN`, `ZS`, `ZC`, `ZW`).

Until those steps are complete, the affected keys remain quarantined.
