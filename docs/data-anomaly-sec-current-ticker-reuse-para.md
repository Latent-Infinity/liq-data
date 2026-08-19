# SEC current-ticker collision for historical `PARA`

**Observed:** 2026-08-19
**Status:** QUARANTINED — the current association is not admissible as
historical Paramount identity evidence.
**Detection context:** non-signal Path D security-identity reconciliation.

## Finding

The SEC `company_tickers.json` lookup currently associates `PARA` with CIK
`0001826011`, title `Banzai International, Inc.`. The Path D PIT membership
window for `PARA` is 2022-02-17 through 2025-07-23, while the FINRA rows inside
that window identify `Paramount Global Class B Common Stock`.

The reconciliation inspected all 70 eligible SEC filing-index entries for the
current CIK inside the PIT window. None of their accession-bound
`dei:TradingSymbol` values matched `PARA`. This is a ticker-reuse collision, not
an absent-file result.

The SEC documents `company_tickers.json` as a periodically updated search
association whose accuracy and scope are not guaranteed. It is current
candidate metadata, not a point-in-time symbology source.

No bars, price values, returns, signals, portfolios, grids, or P&L were read or
computed during detection.

## Immediate disposition

- Do not associate CIK `0001826011` with the historical S&P 500 `PARA` member.
- Keep `PARA` outside deleted-member bar persistence and coverage claims.
- Do not infer the historical Paramount CIK from company-name similarity.
- Preserve the raw SEC candidate, FINRA issue name, and 70 accession checks in
  the write-once identity-evidence artifact.

## Repair requirement

A future repair requires accession-bound evidence discovered from an
independently sourced historical CIK or a frozen corporate-action/symbology
record. It must not use the current SEC ticker lookup alone.
