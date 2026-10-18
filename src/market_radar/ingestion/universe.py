"""The initial, configurable Vietnamese equity research universe."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_type: str
    sector: str
    source: str


DEFAULT_SECTORS: dict[str, str] = {
    "FPT": "Technology",
    "HPG": "Steel",
    "MWG": "Retail",
    "VNM": "Consumer goods",
    "CTG": "Banking",
    "CTS": "Securities",
    "VCB": "Banking",
    "SSI": "Securities",
    "GAS": "Energy",
    "VHM": "Real estate",
}


def build_universe(symbols: list[str] | None = None) -> list[Asset]:
    """Build equities in caller order, followed by the VNINDEX benchmark."""
    requested = symbols if symbols is not None else list(DEFAULT_SECTORS)
    assets: list[Asset] = []
    seen: set[str] = set()
    for value in requested:
        symbol = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{1,12}", symbol):
            raise ValueError(f"Invalid stock symbol: {value!r}")
        if symbol == "VNINDEX":
            raise ValueError("VNINDEX is included automatically as an index")
        if symbol not in seen:
            assets.append(Asset(symbol, "equity", DEFAULT_SECTORS.get(symbol, "Unclassified"), "vci"))
            seen.add(symbol)
    # VCI is a documented Community source for index bars. KBS returned two
    # conflicting bars for one date during the Task 002 live check.
    assets.append(Asset("VNINDEX", "index", "Market benchmark", "vci"))
    return assets
