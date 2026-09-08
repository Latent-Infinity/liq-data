"""Read-only Coinbase Derivatives Exchange instrument metadata.

Coinbase Derivatives Exchange (CDE, historically FairX) uses a different API
host and credential set from Coinbase Exchange spot.  This deliberately narrow
provider exposes only contract identity needed for a non-signal readiness
audit.  Funding history and price data stay out of the interface until a
dataset/fold contract is human-frozen.

API references:
- https://docs.cdp.coinbase.com/api-reference/derivatives-api/rest-api/authentication
- https://docs.cdp.coinbase.com/api-reference/derivatives-api/rest-api/instruments/get-instrument-details
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from liq.data.exceptions import (
    AuthenticationError,
    ProviderError,
    RateLimitError,
    SchemaValidationError,
)


@dataclass(frozen=True)
class CoinbaseDerivativesInstrument:
    """Identity fields required to adjudicate a CDE perpetual contract."""

    symbol: str
    product_code: str
    contract_type: str
    trading_state: str
    contract_size: Decimal
    activation_time: datetime | None
    expiration_time: datetime | None
    is_perp: bool
    funding_interval_minutes: int | None


def _required_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"Coinbase Derivatives instrument missing {field}")
    return value.strip()


def _optional_utc_datetime(payload: Mapping[str, Any], field: str) -> datetime | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"Coinbase Derivatives instrument has invalid {field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SchemaValidationError(f"Coinbase Derivatives instrument has invalid {field}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_decimal(payload: Mapping[str, Any], field: str) -> Decimal:
    value = payload.get(field)
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise SchemaValidationError(f"Coinbase Derivatives instrument has invalid {field}") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise SchemaValidationError(f"Coinbase Derivatives instrument has invalid {field}")
    return parsed


def _parse_instrument(payload: Mapping[str, Any]) -> CoinbaseDerivativesInstrument:
    is_perp = payload.get("is_perp")
    if not isinstance(is_perp, bool):
        raise SchemaValidationError("Coinbase Derivatives instrument missing is_perp")

    funding_interval = payload.get("funding_interval_minutes")
    if funding_interval is not None and (
        isinstance(funding_interval, bool)
        or not isinstance(funding_interval, int)
        or funding_interval <= 0
    ):
        raise SchemaValidationError(
            "Coinbase Derivatives instrument has invalid funding_interval_minutes"
        )
    if is_perp and funding_interval is None:
        raise SchemaValidationError(
            "Coinbase Derivatives perpetual missing funding_interval_minutes"
        )

    return CoinbaseDerivativesInstrument(
        symbol=_required_text(payload, "symbol"),
        product_code=_required_text(payload, "product_code"),
        contract_type=_required_text(payload, "contract_type"),
        trading_state=_required_text(payload, "trading_state"),
        contract_size=_positive_decimal(payload, "contract_size"),
        activation_time=_optional_utc_datetime(payload, "activation_time"),
        expiration_time=_optional_utc_datetime(payload, "expiration_time"),
        is_perp=is_perp,
        funding_interval_minutes=funding_interval,
    )


class CoinbaseDerivativesProvider:
    """Authenticated, read-only CDE contract-identity client."""

    name = "coinbase_derivatives"
    BASE_URL = "https://api.exchange.fairx.net"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        passphrase: str,
        *,
        timeout: float = 30.0,
    ) -> None:
        for field, value in (
            ("api_key", api_key),
            ("api_secret", api_secret),
            ("passphrase", passphrase),
        ):
            if not value:
                raise ValueError(f"{field} must be configured")
        try:
            secret_bytes = base64.b64decode(api_secret, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("api_secret must be valid base64") from exc
        if not secret_bytes:
            raise ValueError("api_secret must decode to non-empty bytes")

        self._api_key = api_key
        self._api_secret = api_secret
        self._secret_bytes = secret_bytes
        self._passphrase = passphrase
        self._timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:  # pragma: no cover - live transport
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def close(self) -> None:
        """Close the lazy HTTP client when one was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def _signature(self, timestamp: str, method: str, endpoint: str, body: str) -> str:
        message = f"{timestamp}{method.upper()}{endpoint}{body}".encode()
        digest = hmac.new(self._secret_bytes, message, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        body = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
            if payload is not None
            else ""
        )
        timestamp = str(int(time.time()))
        headers = {
            "CB-ACCESS-KEY": self._api_key,
            "CB-ACCESS-SIGN": self._signature(timestamp, method, endpoint, body),
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": self._passphrase,
            "Content-Type": "application/json",
        }
        try:
            response = self._get_client().request(
                method,
                f"{self.BASE_URL}{endpoint}",
                content=body,
                headers=headers,
            )
        except httpx.RequestError:
            raise ProviderError("Coinbase Derivatives request failed") from None

        if response.status_code in (401, 403):
            raise AuthenticationError("Coinbase Derivatives authentication or entitlement failed")
        if response.status_code == 429:
            raise RateLimitError("Coinbase Derivatives rate limit exceeded")
        if response.status_code != 200:
            raise ProviderError(
                f"Coinbase Derivatives returned HTTP {response.status_code} for {endpoint}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise SchemaValidationError("Coinbase Derivatives returned invalid JSON") from exc

    def list_instruments(
        self, product_codes: Sequence[str]
    ) -> tuple[CoinbaseDerivativesInstrument, ...]:
        """Return normalized identity metadata for the requested CDE products."""
        if isinstance(product_codes, str):
            raise ValueError("product_codes must be a non-empty sequence")
        normalized = tuple(
            dict.fromkeys(code.strip().upper() for code in product_codes if code.strip())
        )
        if not normalized:
            raise ValueError("product_codes must be a non-empty sequence")

        response = self._request(
            "POST", "/rest/instruments", payload={"product_codes": list(normalized)}
        )
        if not isinstance(response, list):
            raise SchemaValidationError("Coinbase Derivatives instruments response must be a list")
        if any(not isinstance(item, Mapping) for item in response):
            raise SchemaValidationError(
                "Coinbase Derivatives instruments response contains a non-object"
            )
        return tuple(_parse_instrument(item) for item in response)
