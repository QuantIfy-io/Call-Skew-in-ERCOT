"""ERCOT real-time hub LMP (HB_NORTH) — no API key.

Historical yearly RTM hub/zone files: NP6-785-ER (reportTypeId 13061).
Recent 15-minute SPP: NP6-905-CD (reportTypeId 12301), used only to fill
the current-year gap after the weekly archive.
"""
from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HEADERS = {"User-Agent": "QuantIfyDemo/1.0", "Accept": "*/*"}
TIMEOUT = 30
HISTORY_START = pd.Timestamp("2021-01-01")
HUB = "HB_NORTH"

ERCOT_RT_HIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13061"
ERCOT_RT_RECENT = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=12301"
ERCOT_DOWNLOAD = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={doc_id}"

CACHE_DIR = Path(__file__).resolve().parent / "data" / "cache"

# Scarcity / crash regimes used in the study.
NEG_CUT = 0.0
TIGHT_CUT = 250.0
SCARCITY_CUT = 1_000.0


def _to_hourly(ts: pd.Series, price: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"ts": pd.to_datetime(ts), "price": pd.to_numeric(price, errors="coerce")})
    df = df.dropna()
    if df.empty:
        return pd.DataFrame(columns=["ts", "price", "high", "low"])
    df["hour"] = df["ts"].dt.floor("h")
    g = df.groupby("hour")["price"]
    out = pd.DataFrame(
        {"ts": g.mean().index, "price": g.mean().to_numpy(), "high": g.max().to_numpy(), "low": g.min().to_numpy()}
    )
    return out.sort_values("ts").reset_index(drop=True)


def daily_from_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame(
            columns=["date", "mean", "high", "low", "n_neg", "n_tight", "n_scarcity", "hours"]
        )
    h = hourly.copy()
    h["date"] = pd.to_datetime(h["ts"]).dt.normalize()
    high_col = "high" if "high" in h.columns else "price"
    low_col = "low" if "low" in h.columns else "price"
    g = h.groupby("date")
    out = pd.DataFrame(
        {
            "mean": g["price"].mean(),
            "high": g[high_col].max(),
            "low": g[low_col].min(),
            "hours": g["price"].size(),
            "n_neg": g["price"].apply(lambda s: int((s < NEG_CUT).sum())),
            "n_tight": g["price"].apply(lambda s: int((s >= TIGHT_CUT).sum())),
            "n_scarcity": g["price"].apply(lambda s: int((s >= SCARCITY_CUT).sum())),
        }
    ).reset_index()
    return out.sort_values("date")


def classify_price(px: float, cap: float) -> str:
    if px < NEG_CUT:
        return "crash / negative (put)"
    if px >= 0.90 * cap:
        return "at cap (call / ORDC)"
    if px >= SCARCITY_CUT:
        return "scarcity (call / ORDC)"
    if px >= TIGHT_CUT:
        return "tight"
    return "normal"


def _ercot_doc_list(url: str) -> list[dict]:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    payload = r.json()
    raw_list = payload["ListDocsByRptTypeRes"]["DocumentList"]
    docs = []
    for item in raw_list:
        d = item["Document"] if isinstance(item, dict) and "Document" in item else item
        docs.append(d)
    return docs


def _download_zip(doc_id: str) -> bytes:
    url = ERCOT_DOWNLOAD.format(doc_id=doc_id)
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.content


def _frames_from_zip(content: bytes) -> list[pd.DataFrame]:
    frames = []
    with zipfile.ZipFile(BytesIO(content)) as zf:
        for name in zf.namelist():
            lower = name.lower()
            with zf.open(name) as fh:
                blob = fh.read()
            if lower.endswith(".csv"):
                frames.append(pd.read_csv(BytesIO(blob), low_memory=False))
            elif lower.endswith((".xlsx", ".xls")):
                xls = pd.ExcelFile(BytesIO(blob))
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    if _hub_intervals(df).empty:
                        df = pd.read_excel(xls, sheet_name=sheet, skiprows=1)
                    frames.append(df)
    return frames


def _colmap(df: pd.DataFrame) -> dict[str, str]:
    return {str(c).strip().lower().replace(" ", ""): c for c in df.columns}


def _pick(cols: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in cols:
            return cols[key]
    return None


def _hub_intervals(df: pd.DataFrame, hub: str = HUB) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "price"])
    cols = _colmap(df)
    sp_col = _pick(cols, "settlementpoint", "settlementpointname", "settlementpointname")
    px_col = _pick(cols, "settlementpointprice", "spp", "price", "rtspp")
    date_col = _pick(cols, "deliverydate", "operday", "date", "deliverydate")
    hour_col = _pick(cols, "deliveryhour", "hourending", "he", "hour")
    int_col = _pick(cols, "deliveryinterval", "interval", "settlementinterval")
    if sp_col is None or px_col is None or date_col is None:
        return pd.DataFrame(columns=["ts", "price"])
    sp = df[sp_col].astype(str).str.strip().str.upper()
    keep = [date_col, px_col]
    if hour_col:
        keep.append(hour_col)
    if int_col:
        keep.append(int_col)
    sub = df.loc[sp == hub.upper(), keep].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ts", "price"])
    sub["date"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub["price"] = pd.to_numeric(sub[px_col], errors="coerce")
    sub = sub.dropna(subset=["date", "price"])
    if hour_col and hour_col in sub.columns:
        he = pd.to_numeric(sub[hour_col], errors="coerce").fillna(1).astype(int).clip(1, 24)
    else:
        he = pd.Series(1, index=sub.index)
    if int_col and int_col in sub.columns:
        iv = pd.to_numeric(sub[int_col], errors="coerce").fillna(1).astype(int).clip(1, 4)
    else:
        iv = pd.Series(1, index=sub.index)
    minutes = (he - 1) * 60 + (iv - 1) * 15
    sub["ts"] = sub["date"] + pd.to_timedelta(minutes, unit="m")
    return sub[["ts", "price"]].dropna()


def _hourly_from_docs(docs: list[dict], max_workers: int = 4) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def _one(doc: dict) -> pd.DataFrame:
        parts = []
        for df in _frames_from_zip(_download_zip(str(doc["DocID"]))):
            parts.append(_hub_intervals(df))
        if not parts:
            return pd.DataFrame(columns=["ts", "price"])
        return pd.concat(parts, ignore_index=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, d) for d in docs]
        for fut in as_completed(futs):
            try:
                frames.append(fut.result())
            except Exception:
                continue
    if not frames:
        return pd.DataFrame(columns=["ts", "price", "high", "low"])
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna().drop_duplicates(subset=["ts"]).sort_values("ts")
    raw = raw.loc[raw["ts"] >= HISTORY_START]
    return _to_hourly(raw["ts"], raw["price"])


def fetch_rt_historical() -> pd.DataFrame:
    """Yearly RTM hub/zone archive (NP6-785-ER), HB_NORTH only, from HISTORY_START."""
    docs = []
    for d in _ercot_doc_list(ERCOT_RT_HIST):
        friendly = str(d.get("FriendlyName", ""))
        if not friendly.upper().startswith("RTMLZHBSPP_"):
            continue
        try:
            year = int(friendly.split("_")[-1])
        except ValueError:
            continue
        if year < HISTORY_START.year:
            continue
        docs.append(d)
    if not docs:
        raise RuntimeError("No ERCOT RTMLZHBSPP yearly files found")
    hourly = _hourly_from_docs(docs, max_workers=3)
    if hourly.empty:
        raise RuntimeError("ERCOT historical RT hub file parsed empty for HB_NORTH")
    return hourly


def fetch_rt_recent(max_files: int = 96) -> pd.DataFrame:
    """Last ~1 day of NP6-905-CD 15-minute SPP (fills a short gap after the weekly archive)."""
    docs = []
    for d in _ercot_doc_list(ERCOT_RT_RECENT):
        name = str(d.get("FriendlyName", "")).lower()
        constructed = str(d.get("ConstructedName", "")).lower()
        if "csv" not in name and "csv" not in constructed:
            continue
        docs.append(d)
    docs = docs[:max_files]
    if not docs:
        return pd.DataFrame(columns=["ts", "price", "high", "low"])
    return _hourly_from_docs(docs, max_workers=6)


def _cache_path() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / "hb_north_rt_hourly.csv.gz"


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False, compression="gzip")
    except Exception:
        pass


def _read_cache(path: Path) -> pd.DataFrame:
    cached = pd.read_csv(path, compression="gzip")
    cached["ts"] = pd.to_datetime(cached["ts"])
    return cached


def _fallback_hourly() -> pd.DataFrame:
    """Deterministic path with Uri-style cap, summer scarcity, and negative-price dumps."""
    rng = np.random.default_rng(21)
    idx = pd.date_range(HISTORY_START, pd.Timestamp.today().normalize(), freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    seasonal = 28.0 + 18.0 * np.sin(2 * np.pi * (doy - 200) / 365.0)
    diurnal = 8.0 * np.sin(2 * np.pi * (hour - 8) / 24.0)
    z = rng.normal(0.0, 1.0, n)
    z = z + (rng.random(n) < 0.004) * rng.exponential(4.0, n)
    target = np.log(np.maximum(seasonal + diurnal, 8.0))
    # AR(1) around the seasonal log-price path.
    phi = 0.85
    innov = 0.15 * target + 0.12 * z
    logp = np.empty(n)
    logp[0] = np.log(35.0)
    # Vectorized AR(1): y_t = phi y_{t-1} + e_t  via recursive filter.
    # y = lfilter([1], [1, -phi], e) with y0 baked into e[0].
    e = innov.copy()
    e[0] = logp[0]
    for i in range(1, n):
        logp[i] = phi * logp[i - 1] + e[i]
    px = np.exp(logp)
    spring = (idx.month >= 3) & (idx.month <= 5) & (hour >= 10) & (hour <= 16)
    dump = spring & (rng.random(n) < 0.03)
    px = np.where(dump, -rng.uniform(5.0, 35.0, n), px)
    summer = (idx.month >= 7) & (idx.month <= 9) & (hour >= 14) & (hour <= 20)
    hot = summer & (rng.random(n) < 0.012)
    px = np.where(hot, np.minimum(5_000.0, 400.0 + rng.exponential(600.0, n)), px)
    uri = (idx >= "2021-02-14") & (idx < "2021-02-20")
    px = np.where(uri, 9_000.0 - rng.uniform(0.0, 80.0, n), px)
    d22 = (idx >= "2022-12-23") & (idx < "2022-12-25") & (hour >= 6) & (hour <= 21)
    px = np.where(d22, np.minimum(5_000.0, 800.0 + rng.exponential(400.0, n)), px)
    high = np.where(px < 0, px + 5.0, px * (1.0 + rng.uniform(0.0, 0.08, n)))
    low = np.where(px < 0, px - 5.0, px * (1.0 - rng.uniform(0.0, 0.08, n)))
    return pd.DataFrame({"ts": idx, "price": px, "high": high, "low": low})


def fetch_hb_north_rt() -> tuple[pd.DataFrame, str]:
    """Hourly HB_NORTH RT settlement point price from HISTORY_START. Status live | fallback."""
    cache = _cache_path()
    try:
        hist = fetch_rt_historical()
        try:
            recent = fetch_rt_recent()
        except Exception:
            recent = pd.DataFrame(columns=hist.columns)
        combined = pd.concat([hist, recent], ignore_index=True)
        combined["ts"] = pd.to_datetime(combined["ts"])
        combined = combined.dropna(subset=["ts", "price"])
        combined = combined.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
        combined = combined.loc[combined["ts"] >= HISTORY_START].reset_index(drop=True)
        if combined.empty or len(combined) < 24 * 30:
            raise RuntimeError("RT series too short")
        try:
            _write_cache(combined, cache)
        except Exception:
            pass
        return combined, "live"
    except Exception:
        if cache.exists():
            try:
                cached = _read_cache(cache)
                if len(cached) >= 24 * 30:
                    return cached, "cache"
            except Exception:
                pass
        return _fallback_hourly(), "fallback"


def event_windows(hourly: pd.DataFrame) -> pd.DataFrame:
    """Named scarcity (call) and crash (put) windows in the sample."""
    rows = [
        ("Winter Storm Uri", "2021-02-14", "2021-02-20", "call", "Supply freeze; HCAP $9,000; ORDC fully engaged"),
        ("Summer 2022 tightness", "2022-07-10", "2022-07-20", "call", "Hot-weather ORDC / cap-adjacent prints"),
        ("Dec 2022 freeze", "2022-12-22", "2022-12-25", "call", "Winter scarcity — still a short-call event"),
        ("Summer 2023 scarcity", "2023-08-15", "2023-08-25", "call", "Record load; RT into four figures"),
        ("Shoulder negative dump", "2024-04-01", "2024-04-21", "put", "Midday wind/solar oversupply; demand-side put"),
        ("Mild-demand crash sample", "2025-03-15", "2025-04-05", "put", "Shoulder crash / freeze-off of industrial load"),
    ]
    out = []
    h = hourly.copy()
    h["ts"] = pd.to_datetime(h["ts"])
    for name, a, b, side, note in rows:
        sl = h.loc[(h["ts"] >= a) & (h["ts"] < b)]
        if sl.empty:
            continue
        out.append(
            {
                "Event": name,
                "Side": side,
                "Start": pd.Timestamp(a).date(),
                "End": pd.Timestamp(b).date(),
                "Hours": int(len(sl)),
                "Max RT": float(sl["price"].max()),
                "Min RT": float(sl["price"].min()),
                "Mean RT": float(sl["price"].mean()),
                "Hours < $0": int((sl["price"] < 0).sum()),
                "Hours ≥ $1,000": int((sl["price"] >= SCARCITY_CUT).sum()),
                "Note": note,
            }
        )
    return pd.DataFrame(out)
