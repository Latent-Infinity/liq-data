import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import respx

from liq.data.providers.tradestation import TradeStationProvider


@respx.mock
def test_minute_pagination_requests_previous_complete_minute() -> None:
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/tradestation_pagination_boundary.json").read_text()
    )
    records = {row["TimeStamp"]: row for row in fixture["Bars"]}
    boundary = records["2024-05-30T14:11:00Z"]
    previous = records["2024-05-30T14:10:00Z"]
    provider = TradeStationProvider(client_id="test", client_secret="test", refresh_token="test")
    respx.post("https://signin.tradestation.com/oauth/token").respond(
        200, json={"access_token": "test", "expires_in": 1200}
    )
    requested: list[str] = []

    def response(request: httpx.Request) -> httpx.Response:
        lastdate = request.url.params["lastdate"]
        requested.append(lastdate)
        if len(requested) == 1:
            return httpx.Response(200, json={"Bars": [boundary]})
        if len(requested) == 2:
            # The captured provider response rounded 14:10:59 up to 14:11.
            selected = [previous, boundary] if lastdate.endswith("59Z") else [previous]
            return httpx.Response(200, json={"Bars": selected})
        return httpx.Response(200, json={"Bars": []})

    respx.get(url__regex=r".*/marketdata/barcharts/TSM").mock(side_effect=response)

    result = provider.fetch_bars("TSM", date(2024, 5, 30), date(2024, 5, 30), "1m")

    assert result.height == result["timestamp"].n_unique() == 2
    assert requested[1] == "2024-05-30T14:10:00Z"
    assert result["timestamp"].min() == datetime(2024, 5, 30, 14, 10, tzinfo=UTC)
