import json
import math
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import numpy as np
import pandas as pd
import yfinance as yf
from gspread.utils import rowcol_to_a1
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

JST = ZoneInfo("Asia/Tokyo")
NOW_JST = datetime.now(JST)
TODAY_JST = NOW_JST.strftime("%Y-%m-%d")
IS_SUNDAY_JST = NOW_JST.weekday() == 6

APP_SECRET_ENV = "APP_SECRET_JSON"
MIN_TRAIN_N = 300
DATA_QUALITY_THRESHOLD = 0.70
ALPHAS = [0.1, 1, 10, 100, 1000]
RANDOM_STATE = 42
MAX_RETRIES = 2
RETRY_SLEEP_SECONDS = 2
PROGRESS_EVERY = 100
MARKET_CAP_DIVISOR = 100_000_000
PRICE_BATCH_SIZE = 8
MAX_WORKERS = 4  # 8→4に削減してCrumb失効を緩和
ENRICH_BATCH_SIZE = 200  # バッチサイズ: この件数ごとにセッションをリセット
CACHE_TTL_DAYS = 7
CACHE_START_COL = 20  # T列

EXCLUDED_SECTORS = {
    "銀行業",
    "証券、商品先物取引業",
    "保険業",
    "その他金融業",
}

OUTPUT_COLUMNS = [
    "Price",
    "ActualMarketCap",
    "PredictedMarketCap",
    "ResidualLog",
    "MispricingRatio",
    "DataQuality",
    "OverallRank",
    "SectorRank",
    "FinancialAsOf",
    "QuoteAsOf",
    "ModelR2",
    "ModelMAE",
    "TrainN",
    "Notes",
    "Status",
    "Error",
]

RAW_FEATURE_SPECS = {
    "total_assets": ["Total Assets"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "equity": ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    "cash": [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Operating Income Loss"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "operating_cf": [
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
        "Net Cash Provided By Operating Activities",
    ],
    "capex": ["Capital Expenditure", "Capital Expenditure Reported"],
    "dividends_paid": ["Cash Dividends Paid", "Common Stock Dividend Paid", "Dividends Paid"],
    "intangible_assets": [
        "Other Intangible Assets",
        "Intangible Assets",
        "Goodwill And Other Intangible Assets",
    ],
}

RAW_FEATURE_KEYS = list(RAW_FEATURE_SPECS.keys())
MODEL_NUMERIC_FEATURES = RAW_FEATURE_KEYS + ["free_cf"]
LOG1P_FEATURES = [
    "total_assets",
    "total_liabilities",
    "cash",
    "current_assets",
    "current_liabilities",
    "revenue",
    "intangible_assets",
]
ZERO_FILL_FEATURES = {"capex", "dividends_paid"}
WARNING_SECTOR = "不動産業"

BS_FEATURE_KEYS = [
    "total_assets",
    "total_liabilities",
    "equity",
    "cash",
    "current_assets",
    "current_liabilities",
    "intangible_assets",
]
PL_FEATURE_KEYS = [
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
]
CF_FEATURE_KEYS = [
    "operating_cf",
    "capex",
    "dividends_paid",
]
MIN_FEATURE_COUNT = math.ceil(DATA_QUALITY_THRESHOLD * len(RAW_FEATURE_KEYS))

CACHE_HEADERS = [
    "CacheCode",
    "CacheUpdatedAt",
    "CacheFinancialAsOf",
    "CacheDataQuality",
    "CacheFreeCf",
] + [f"Cache_{key}" for key in RAW_FEATURE_KEYS]


def log(msg: str) -> None:
    print(msg, flush=True)


def safe_exc_name(exc: Exception) -> str:
    return type(exc).__name__


def summarize_status_counts(rows: list[dict]) -> str:
    counts = Counter(row.get("status", "") for row in rows)
    keys = ["OK", "除外業種", "データ不足", "入力不正", "学習失敗"]
    parts = [f"{k}={counts.get(k, 0)}" for k in keys if counts.get(k, 0) > 0]
    return ", ".join(parts) if parts else "status=0"


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\u3000", " ")
    text = re.sub(r"\s+", "", text)
    return text


def normalize_label(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def is_valid_code(value: str) -> bool:
    code = str(value).strip().upper()
    return bool(re.fullmatch(r"(?:\d{4}|\d{3}[A-Z])", code))


def load_config() -> dict:
    raw = os.environ.get(APP_SECRET_ENV)
    if not raw:
        raise RuntimeError(f"Environment variable '{APP_SECRET_ENV}' is not set.")

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{APP_SECRET_ENV} is not valid JSON.") from exc

    required = ["sheet_url", "worksheet_name", "gcp_service_account_json"]
    missing = [k for k in required if k not in config]
    if missing:
        raise RuntimeError(f"Missing keys in {APP_SECRET_ENV}: {', '.join(missing)}")

    return config


def open_worksheet(config: dict):
    gc = gspread.service_account_from_dict(config["gcp_service_account_json"])
    sh = gc.open_by_url(config["sheet_url"])
    return sh.worksheet(config["worksheet_name"])


def get_cell(raw_row: list[str], idx: int) -> str:
    if idx < len(raw_row):
        return raw_row[idx].strip()
    return ""


def parse_optional_float(value: str):
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def parse_cache_from_row(raw_row: list[str]) -> dict | None:
    start_idx = CACHE_START_COL - 1
    cache_code = get_cell(raw_row, start_idx)
    cache_updated_at = get_cell(raw_row, start_idx + 1)
    cache_financial_as_of = get_cell(raw_row, start_idx + 2)
    cache_data_quality = parse_optional_float(get_cell(raw_row, start_idx + 3))
    cache_free_cf = parse_optional_float(get_cell(raw_row, start_idx + 4))

    raw_values = {}
    any_value = False
    for i, key in enumerate(RAW_FEATURE_KEYS):
        value = parse_optional_float(get_cell(raw_row, start_idx + 5 + i))
        raw_values[key] = value
        if value is not None:
            any_value = True

    if not cache_code and not cache_updated_at and not any_value and cache_free_cf is None:
        return None

    return {
        "code": cache_code,
        "updated_at": cache_updated_at,
        "financial_as_of": cache_financial_as_of,
        "data_quality": cache_data_quality,
        "free_cf": cache_free_cf,
        "raw_values": raw_values,
    }


def read_input_rows(ws) -> list[dict]:
    values = ws.get_all_values()
    if not values:
        return []

    rows = []
    for idx, raw_row in enumerate(values[1:], start=2):
        code = get_cell(raw_row, 0)
        name = get_cell(raw_row, 1)
        sector = get_cell(raw_row, 2)

        if not code and not name and not sector:
            continue

        rows.append(
            {
                "sheet_row": idx,
                "code": code,
                "name": name,
                "sector": sector,
                "status": "",
                "error": "",
                "notes": "",
                "cache": parse_cache_from_row(raw_row),
            }
        )

    return rows


def reset_yfinance_session() -> None:
    """yfinanceのグローバルセッションとCrumbキャッシュをリセットする。"""
    try:
        yf.utils.get_yf_logger()  # ロガー初期化（副作用なし、import確認用）
    except Exception:  # noqa: BLE001
        pass
    try:
        # yfinance内部のセッションキャッシュをクリアする
        if hasattr(yf.utils, "_curlfile"):
            yf.utils._curlfile = None
        cache = getattr(yf.utils, "_CRUMB_CACHE", None)
        if cache is not None and hasattr(cache, "clear"):
            cache.clear()
        # requests_cache使用時はセッション自体を再生成
        session = getattr(yf, "_requests_cache_session", None)
        if session is not None and hasattr(session, "close"):
            session.close()
        yf.utils.requests = __import__("requests")
    except Exception:  # noqa: BLE001
        pass


def call_with_retry(func, *args, **kwargs):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < MAX_RETRIES:
                # 401(Crumb失効)の場合はセッションリセットを挟む
                exc_str = str(exc)
                if "401" in exc_str or "Unauthorized" in exc_str or "Invalid Crumb" in exc_str:
                    reset_yfinance_session()
                time.sleep(RETRY_SLEEP_SECONDS)
    raise last_exc


def extract_latest_price(history_df: pd.DataFrame) -> tuple[float | None, str | None]:
    if history_df is None or history_df.empty or "Close" not in history_df.columns:
        return None, None

    close_series = history_df["Close"].dropna()
    if close_series.empty:
        return None, None

    last_idx = close_series.index[-1]
    price = float(close_series.iloc[-1])
    price_date = pd.Timestamp(last_idx).strftime("%Y-%m-%d")
    return price, price_date


def chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def fetch_batch_prices(codes: list[str]) -> dict[str, tuple[float | None, str]]:
    results = {code: (None, "") for code in codes}
    if not codes:
        return results

    tickers = [f"{code}.T" for code in codes]

    try:
        batch_df = call_with_retry(
            yf.download,
            tickers=tickers,
            period="5d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception:  # noqa: BLE001
        return results

    if batch_df is None or batch_df.empty:
        return results

    if isinstance(batch_df.columns, pd.MultiIndex):
        top_level = set(str(x) for x in batch_df.columns.get_level_values(0))
        for code in codes:
            ticker_key = f"{code}.T"
            if ticker_key in top_level:
                sub_df = batch_df[ticker_key]
            elif code in top_level:
                sub_df = batch_df[code]
            else:
                continue
            price, dt = extract_latest_price(sub_df)
            results[code] = (price, dt or "")
    else:
        if len(codes) == 1:
            price, dt = extract_latest_price(batch_df)
            results[codes[0]] = (price, dt or "")

    return results


def fetch_all_batch_prices(codes: list[str]) -> dict[str, tuple[float | None, str]]:
    price_map = {}
    for code_chunk in chunked(codes, PRICE_BATCH_SIZE):
        price_map.update(fetch_batch_prices(code_chunk))
    return price_map


def pick_info_number(info: dict, keys: list[str]) -> float | None:
    if not isinstance(info, dict):
        return None
    for key in keys:
        value = info.get(key)
        if value is None:
            continue
        try:
            if pd.notna(value):
                return float(value)
        except Exception:  # noqa: BLE001
            continue
    return None


def statement_value(df: pd.DataFrame, candidates: list[str]) -> tuple[float | None, str | None]:
    if df is None or df.empty:
        return None, None

    index_map = {normalize_label(idx): idx for idx in df.index}

    for candidate in candidates:
        key = normalize_label(candidate)
        if key not in index_map:
            continue

        series = df.loc[index_map[key]]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[0]
        series = series.dropna()
        if series.empty:
            continue

        try:
            sorted_index = sorted(series.index, reverse=True)
            for col in sorted_index:
                value = series[col]
                if pd.notna(value):
                    date_str = pd.Timestamp(col).strftime("%Y-%m-%d")
                    return float(value), date_str
        except Exception:  # noqa: BLE001
            value = series.iloc[0]
            if pd.notna(value):
                return float(value), None

    return None, None


def choose_financial_as_of(date_map: dict[str, str | None]) -> str:
    dates = [v for v in date_map.values() if v]
    if not dates:
        return ""
    counter = Counter(dates)
    return counter.most_common(1)[0][0]


def count_available_features(raw_values: dict) -> int:
    return sum(raw_values.get(k) is not None for k in RAW_FEATURE_KEYS)


def build_financial_payload(raw_values: dict, raw_dates: dict) -> dict:
    operating_cf = raw_values.get("operating_cf")
    capex = raw_values.get("capex")
    free_cf = None
    if operating_cf is not None and capex is not None:
        if capex < 0:
            free_cf = operating_cf + capex
        else:
            free_cf = operating_cf - capex

    data_quality_count = count_available_features(raw_values)
    data_quality = data_quality_count / len(RAW_FEATURE_KEYS)

    return {
        "financial_as_of": choose_financial_as_of(raw_dates),
        "data_quality": data_quality,
        "raw_values": raw_values,
        "free_cf": free_cf,
    }


def build_cache_write(code: str, financial_payload: dict, updated_at: str) -> dict:
    return {
        "code": code,
        "updated_at": updated_at,
        "financial_as_of": financial_payload.get("financial_as_of", ""),
        "data_quality": financial_payload.get("data_quality"),
        "free_cf": financial_payload.get("free_cf"),
        "raw_values": dict(financial_payload.get("raw_values", {})),
    }


def has_cache_data(cache_payload: dict | None) -> bool:
    if not cache_payload:
        return False
    if cache_payload.get("free_cf") is not None:
        return True
    raw_values = cache_payload.get("raw_values", {})
    return any(raw_values.get(k) is not None for k in RAW_FEATURE_KEYS)


def is_cache_fresh(updated_at: str) -> bool:
    if not updated_at:
        return False
    try:
        updated_date = datetime.strptime(updated_at, "%Y-%m-%d").date()
    except Exception:  # noqa: BLE001
        return False
    delta_days = (NOW_JST.date() - updated_date).days
    return 0 <= delta_days <= CACHE_TTL_DAYS


def normalize_cache_for_code(code: str, cache_payload: dict | None) -> dict | None:
    if not cache_payload:
        return None
    cache_code = str(cache_payload.get("code", "")).strip().upper()
    if cache_code != code:
        return None
    return cache_payload


def fetch_financial_payload(ticker) -> dict:
    raw_values = {key: None for key in RAW_FEATURE_KEYS}
    raw_dates = {key: None for key in RAW_FEATURE_KEYS}

    bs = call_with_retry(lambda: ticker.balance_sheet)
    for feature in BS_FEATURE_KEYS:
        value, dt = statement_value(bs, RAW_FEATURE_SPECS[feature])
        raw_values[feature] = value
        raw_dates[feature] = dt

    current_count = count_available_features(raw_values)
    if current_count + len(PL_FEATURE_KEYS) + len(CF_FEATURE_KEYS) < MIN_FEATURE_COUNT:
        return build_financial_payload(raw_values, raw_dates)

    fin = call_with_retry(lambda: ticker.financials)
    for feature in PL_FEATURE_KEYS:
        value, dt = statement_value(fin, RAW_FEATURE_SPECS[feature])
        raw_values[feature] = value
        raw_dates[feature] = dt

    current_count = count_available_features(raw_values)
    if current_count + len(CF_FEATURE_KEYS) < MIN_FEATURE_COUNT:
        return build_financial_payload(raw_values, raw_dates)

    cf = call_with_retry(lambda: ticker.cashflow)
    for feature in CF_FEATURE_KEYS:
        value, dt = statement_value(cf, RAW_FEATURE_SPECS[feature])
        raw_values[feature] = value
        raw_dates[feature] = dt

    return build_financial_payload(raw_values, raw_dates)


def fetch_ticker_payload(
    code: str,
    price_hint: float | None = None,
    quote_as_of_hint: str = "",
    cache_payload: dict | None = None,
    force_refresh: bool = False,
) -> dict:
    code = str(code).strip().upper()
    cache_payload = normalize_cache_for_code(code, cache_payload)

    ticker = yf.Ticker(f"{code}.T")

    price = price_hint
    quote_as_of = quote_as_of_hint or ""

    if price is None:
        history_df = call_with_retry(ticker.history, period="5d", auto_adjust=False)
        price, quote_as_of = extract_latest_price(history_df)

    info = {}
    try:
        info = call_with_retry(lambda: ticker.info)
    except Exception:  # noqa: BLE001
        info = {}

    if price is None:
        price = pick_info_number(info, ["currentPrice", "regularMarketPrice", "previousClose"])
        quote_as_of = TODAY_JST if price is not None else ""

    shares = pick_info_number(info, ["sharesOutstanding"])
    actual_market_cap = pick_info_number(info, ["marketCap"])
    if actual_market_cap is None and price is not None and shares is not None:
        actual_market_cap = float(price) * float(shares)

    if cache_payload and not force_refresh and is_cache_fresh(cache_payload.get("updated_at", "")) and has_cache_data(cache_payload):
        cached_financial_payload = {
            "financial_as_of": cache_payload.get("financial_as_of", ""),
            "data_quality": cache_payload.get("data_quality", 0.0) or 0.0,
            "raw_values": dict(cache_payload.get("raw_values", {})),
            "free_cf": cache_payload.get("free_cf"),
        }
        return {
            "price": price,
            "actual_market_cap": actual_market_cap,
            "quote_as_of": quote_as_of or "",
            "financial_as_of": cached_financial_payload["financial_as_of"],
            "data_quality": cached_financial_payload["data_quality"],
            "raw_values": cached_financial_payload["raw_values"],
            "free_cf": cached_financial_payload["free_cf"],
            "cache_write": cache_payload,
        }

    if actual_market_cap is None or actual_market_cap <= 0:
        if cache_payload and has_cache_data(cache_payload):
            cached_financial_payload = {
                "financial_as_of": cache_payload.get("financial_as_of", ""),
                "data_quality": cache_payload.get("data_quality", 0.0) or 0.0,
                "raw_values": dict(cache_payload.get("raw_values", {})),
                "free_cf": cache_payload.get("free_cf"),
            }
            return {
                "price": price,
                "actual_market_cap": actual_market_cap,
                "quote_as_of": quote_as_of or "",
                "financial_as_of": cached_financial_payload["financial_as_of"],
                "data_quality": cached_financial_payload["data_quality"],
                "raw_values": cached_financial_payload["raw_values"],
                "free_cf": cached_financial_payload["free_cf"],
                "cache_write": cache_payload,
            }

        empty_raw_values = {key: None for key in RAW_FEATURE_KEYS}
        return {
            "price": price,
            "actual_market_cap": actual_market_cap,
            "quote_as_of": quote_as_of or "",
            "financial_as_of": "",
            "data_quality": 0.0,
            "raw_values": empty_raw_values,
            "free_cf": None,
            "cache_write": cache_payload if cache_payload else None,
        }

    try:
        financial_payload = fetch_financial_payload(ticker)
        cache_write = None
        if has_cache_data({"raw_values": financial_payload["raw_values"], "free_cf": financial_payload["free_cf"]}):
            cache_write = build_cache_write(code, financial_payload, TODAY_JST)

        return {
            "price": price,
            "actual_market_cap": actual_market_cap,
            "quote_as_of": quote_as_of or "",
            "financial_as_of": financial_payload["financial_as_of"],
            "data_quality": financial_payload["data_quality"],
            "raw_values": financial_payload["raw_values"],
            "free_cf": financial_payload["free_cf"],
            "cache_write": cache_write,
        }
    except Exception:  # noqa: BLE001
        if cache_payload and has_cache_data(cache_payload):
            cached_financial_payload = {
                "financial_as_of": cache_payload.get("financial_as_of", ""),
                "data_quality": cache_payload.get("data_quality", 0.0) or 0.0,
                "raw_values": dict(cache_payload.get("raw_values", {})),
                "free_cf": cache_payload.get("free_cf"),
            }
            return {
                "price": price,
                "actual_market_cap": actual_market_cap,
                "quote_as_of": quote_as_of or "",
                "financial_as_of": cached_financial_payload["financial_as_of"],
                "data_quality": cached_financial_payload["data_quality"],
                "raw_values": cached_financial_payload["raw_values"],
                "free_cf": cached_financial_payload["free_cf"],
                "cache_write": cache_payload,
            }
        raise


def to_float_or_nan(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return np.nan


def to_oku_or_nan(value):
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(value) / MARKET_CAP_DIVISOR
    except Exception:  # noqa: BLE001
        return np.nan


def build_dataframe(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        rec = {
            "sheet_row": row["sheet_row"],
            "code": row["code"],
            "name": row["name"],
            "sector": row["sector"],
            "status": row["status"],
            "error": row["error"],
            "notes": row["notes"],
            "price": to_float_or_nan(row.get("price")),
            "actual_market_cap": to_float_or_nan(row.get("actual_market_cap")),
            "quote_as_of": row.get("quote_as_of", ""),
            "financial_as_of": row.get("financial_as_of", ""),
            "data_quality": to_float_or_nan(row.get("data_quality")),
        }

        for feature in RAW_FEATURE_KEYS:
            rec[feature] = to_float_or_nan(row.get(feature))
        rec["free_cf"] = to_float_or_nan(row.get("free_cf"))
        records.append(rec)

    return pd.DataFrame(records)


def apply_imputation(train_df: pd.DataFrame, full_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    full = full_df.copy()

    for col in feature_cols:
        if col in ZERO_FILL_FEATURES:
            train[col] = train[col].fillna(0.0)
            full[col] = full[col].fillna(0.0)
            continue

        sector_medians = train.groupby("sector", dropna=False)[col].median()
        train[col] = train[col].fillna(train["sector"].map(sector_medians))
        full[col] = full[col].fillna(full["sector"].map(sector_medians))

        global_median = train[col].median()
        train[col] = train[col].fillna(global_median)
        full[col] = full[col].fillna(global_median)

    return train, full


def winsorize_by_train(train_df: pd.DataFrame, full_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    full = full_df.copy()
    for col in feature_cols:
        lower = train[col].quantile(0.01)
        upper = train[col].quantile(0.99)
        train[col] = train[col].clip(lower=lower, upper=upper)
        full[col] = full[col].clip(lower=lower, upper=upper)
    return train, full


def transform_features(train_df: pd.DataFrame, full_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    full = full_df.copy()

    for col in LOG1P_FEATURES:
        if col in feature_cols:
            train[col] = train[col].clip(lower=0).apply(np.log1p)
            full[col] = full[col].clip(lower=0).apply(np.log1p)

    return train, full


def standardize_by_train(train_df: pd.DataFrame, full_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_df.copy()
    full = full_df.copy()
    for col in feature_cols:
        mean = train[col].mean()
        std = train[col].std(ddof=0)
        if std == 0 or pd.isna(std):
            std = 1.0
        train[col] = (train[col] - mean) / std
        full[col] = (full[col] - mean) / std
    return train, full


def make_design_matrix(train_base: pd.DataFrame, pred_base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    numeric_cols = MODEL_NUMERIC_FEATURES.copy()

    train_num, pred_num = apply_imputation(train_base, pred_base, numeric_cols)
    train_num, pred_num = winsorize_by_train(train_num, pred_num, numeric_cols)
    train_num, pred_num = transform_features(train_num, pred_num, numeric_cols)
    train_num, pred_num = standardize_by_train(train_num, pred_num, numeric_cols)

    sector_categories = sorted(train_num["sector"].dropna().astype(str).unique().tolist())
    train_sector = pd.Categorical(train_num["sector"].astype(str), categories=sector_categories)
    pred_sector = pd.Categorical(pred_num["sector"].astype(str), categories=sector_categories)

    train_sector_df = pd.get_dummies(train_sector, prefix="sector", dtype=float)
    pred_sector_df = pd.get_dummies(pred_sector, prefix="sector", dtype=float)
    pred_sector_df = pred_sector_df.reindex(columns=train_sector_df.columns, fill_value=0.0)

    x_train = pd.concat([train_num[numeric_cols].reset_index(drop=True), train_sector_df.reset_index(drop=True)], axis=1)
    x_pred = pd.concat([pred_num[numeric_cols].reset_index(drop=True), pred_sector_df.reset_index(drop=True)], axis=1)
    return x_train, x_pred, numeric_cols + train_sector_df.columns.tolist()


def select_best_alpha(x_train: pd.DataFrame, y_train: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_alpha = None
    best_score = None

    for alpha in ALPHAS:
        fold_maes = []
        for tr_idx, va_idx in kf.split(x_train):
            model = Ridge(alpha=alpha)
            model.fit(x_train.iloc[tr_idx], y_train[tr_idx])
            pred = model.predict(x_train.iloc[va_idx])
            fold_maes.append(mean_absolute_error(y_train[va_idx], pred))
        avg_mae = float(np.mean(fold_maes))
        if best_score is None or avg_mae < best_score:
            best_score = avg_mae
            best_alpha = alpha
    return float(best_alpha)


def compute_oof_metrics(x_train: pd.DataFrame, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros(len(y_train), dtype=float)
    for tr_idx, va_idx in kf.split(x_train):
        model = Ridge(alpha=alpha)
        model.fit(x_train.iloc[tr_idx], y_train[tr_idx])
        oof_pred[va_idx] = model.predict(x_train.iloc[va_idx])
    return oof_pred, float(r2_score(y_train, oof_pred)), float(mean_absolute_error(y_train, oof_pred))


def assign_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["overall_rank"] = np.nan
    ranked["sector_rank"] = np.nan

    pred_mask = ranked["predicted_market_cap"].notna() & ranked["residual_log"].notna()
    pred_df = ranked.loc[pred_mask].copy()

    if pred_df.empty:
        return ranked

    overall_sorted = pred_df.sort_values(
        by=["residual_log", "mispricing_ratio", "code"],
        ascending=[False, False, True],
    )
    overall_map = {idx: rank for rank, idx in enumerate(overall_sorted.index.tolist(), start=1)}
    ranked.loc[pred_mask, "overall_rank"] = ranked.loc[pred_mask].index.map(overall_map)

    sector_rank_map = {}
    for sector, group in pred_df.groupby("sector", dropna=False):
        sector_sorted = group.sort_values(
            by=["residual_log", "mispricing_ratio", "code"],
            ascending=[False, False, True],
        )
        for rank, idx in enumerate(sector_sorted.index.tolist(), start=1):
            sector_rank_map[idx] = rank

    ranked.loc[pred_mask, "sector_rank"] = ranked.loc[pred_mask].index.map(sector_rank_map)
    return ranked


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:  # noqa: BLE001
        pass
    return value


def dataframe_to_sheet_rows(df: pd.DataFrame) -> list[list]:
    rows = []
    for _, row in df.iterrows():
        rows.append([
            format_cell(row.get("price")),
            format_cell(to_oku_or_nan(row.get("actual_market_cap"))),
            format_cell(to_oku_or_nan(row.get("predicted_market_cap"))),
            format_cell(row.get("residual_log")),
            format_cell(row.get("mispricing_ratio")),
            format_cell(row.get("data_quality")),
            format_cell(row.get("overall_rank")),
            format_cell(row.get("sector_rank")),
            format_cell(row.get("financial_as_of")),
            format_cell(row.get("quote_as_of")),
            format_cell(row.get("model_r2")),
            format_cell(row.get("model_mae")),
            format_cell(row.get("train_n")),
            format_cell(row.get("notes")),
            format_cell(row.get("status")),
            format_cell(row.get("error")),
        ])
    return rows


def cache_to_sheet_rows(rows: list[dict]) -> list[list]:
    out_rows = []
    for row in rows:
        cache_write = row.get("cache_write")
        if not has_cache_data(cache_write):
            out_rows.append([""] * len(CACHE_HEADERS))
            continue

        raw_values = cache_write.get("raw_values", {})
        out_rows.append([
            format_cell(cache_write.get("code")),
            format_cell(cache_write.get("updated_at")),
            format_cell(cache_write.get("financial_as_of")),
            format_cell(cache_write.get("data_quality")),
            format_cell(cache_write.get("free_cf")),
            *[format_cell(raw_values.get(key)) for key in RAW_FEATURE_KEYS],
        ])
    return out_rows


def enrich_row(
    row: dict,
    excluded_sector_norms: set[str],
    price_map: dict[str, tuple[float | None, str]],
    force_refresh_cache: bool,
) -> dict:
    row = dict(row)
    code = str(row["code"]).strip().upper()
    row["code"] = code
    sector = row["sector"]
    normalized_sector = normalize_text(sector)
    row["cache_write"] = row.get("cache")

    if not is_valid_code(code):
        row["status"] = "入力不正"
        row["error"] = "A列が4桁数字または3桁+英字ではありません"
        row["cache_write"] = None
        return row

    if normalized_sector in excluded_sector_norms:
        row["status"] = "除外業種"
        row["error"] = "金融系業種のため学習対象外"
    else:
        row["status"] = "取得中"

    if normalized_sector == normalize_text(WARNING_SECTOR):
        row["notes"] = "注意業種"

    price_hint, quote_hint = price_map.get(code, (None, ""))

    try:
        payload = fetch_ticker_payload(
            code=code,
            price_hint=price_hint,
            quote_as_of_hint=quote_hint,
            cache_payload=row.get("cache"),
            force_refresh=force_refresh_cache,
        )
        row["price"] = payload["price"]
        row["actual_market_cap"] = payload["actual_market_cap"]
        row["quote_as_of"] = payload["quote_as_of"]
        row["financial_as_of"] = payload["financial_as_of"]
        row["data_quality"] = payload["data_quality"]
        for feature in RAW_FEATURE_KEYS:
            row[feature] = payload["raw_values"].get(feature)
        row["free_cf"] = payload["free_cf"]
        row["cache_write"] = payload.get("cache_write")
    except Exception as exc:  # noqa: BLE001
        if row["status"] != "除外業種":
            row["status"] = "データ不足"
            row["error"] = f"yfinance取得失敗: {safe_exc_name(exc)}"
        return row

    if row["status"] != "除外業種":
        if row.get("actual_market_cap") is None or row.get("actual_market_cap", 0) <= 0:
            row["status"] = "データ不足"
            row["error"] = "実時価総額を取得できません"
        elif row.get("data_quality", 0) < DATA_QUALITY_THRESHOLD:
            row["status"] = "データ不足"
            row["error"] = f"DataQuality不足(<{DATA_QUALITY_THRESHOLD:.2f})"
        else:
            row["status"] = "OK"
            row["error"] = ""

    return row


def main() -> None:
    config = load_config()
    ws = open_worksheet(config)

    input_rows = read_input_rows(ws)
    if not input_rows:
        log("INFO start rows=0 result=no_input")
        return

    total_rows = len(input_rows)
    log(f"INFO start rows={total_rows}")

    valid_codes = []
    for row in input_rows:
        code = str(row["code"]).strip().upper()
        if is_valid_code(code):
            valid_codes.append(code)

    log(f"INFO price_batch start codes={len(valid_codes)} batch_size={PRICE_BATCH_SIZE}")
    price_map = fetch_all_batch_prices(valid_codes)
    log("INFO price_batch done")

    excluded_sector_norms = {normalize_text(x) for x in EXCLUDED_SECTORS}
    enriched_rows = []
    processed = 0

    # ENRICH_BATCH_SIZE件ごとにバッチを区切り、バッチ間でセッションをリセットする
    for batch_start in range(0, total_rows, ENRICH_BATCH_SIZE):
        batch = input_rows[batch_start:batch_start + ENRICH_BATCH_SIZE]

        if batch_start > 0:
            reset_yfinance_session()
            log(f"INFO session reset at row={batch_start}")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    enrich_row,
                    row,
                    excluded_sector_norms,
                    price_map,
                    IS_SUNDAY_JST,
                )
                for row in batch
            ]

            for future in futures:
                enriched_rows.append(future.result())
                processed += 1
                if processed % PROGRESS_EVERY == 0 or processed == total_rows:
                    log(f"INFO fetch {processed}/{total_rows} {summarize_status_counts(enriched_rows)}")

    df = build_dataframe(enriched_rows)

    ok_mask = df["status"] == "OK"
    train_df = df.loc[ok_mask].copy().reset_index(drop=False)

    model_r2 = np.nan
    model_mae = np.nan
    train_n = int(len(train_df))

    if train_n >= MIN_TRAIN_N:
        log(f"INFO train start train_n={train_n}")

        train_df["sector"] = train_df["sector"].astype(str)
        x_train, x_pred, _ = make_design_matrix(train_df, train_df)
        y_train = np.log(train_df["actual_market_cap"].astype(float).values)

        best_alpha = select_best_alpha(x_train, y_train)
        _, model_r2, model_mae = compute_oof_metrics(x_train, y_train, best_alpha)

        final_model = Ridge(alpha=best_alpha)
        final_model.fit(x_train, y_train)
        pred_log = final_model.predict(x_pred)

        train_df["predicted_log_mcap"] = pred_log
        train_df["predicted_market_cap"] = np.exp(pred_log)
        train_df["residual_log"] = train_df["predicted_log_mcap"] - np.log(train_df["actual_market_cap"].astype(float))
        train_df["mispricing_ratio"] = (
            train_df["predicted_market_cap"] - train_df["actual_market_cap"]
        ) / train_df["actual_market_cap"]

        train_df = assign_ranks(train_df)

        df["predicted_market_cap"] = np.nan
        df["residual_log"] = np.nan
        df["mispricing_ratio"] = np.nan
        df["overall_rank"] = np.nan
        df["sector_rank"] = np.nan

        for _, r in train_df.iterrows():
            original_idx = int(r["index"])
            df.at[original_idx, "predicted_market_cap"] = float(r["predicted_market_cap"])
            df.at[original_idx, "residual_log"] = float(r["residual_log"])
            df.at[original_idx, "mispricing_ratio"] = float(r["mispricing_ratio"])
            df.at[original_idx, "overall_rank"] = float(r["overall_rank"])
            df.at[original_idx, "sector_rank"] = float(r["sector_rank"])

        log(
            f"INFO train done train_n={train_n} alpha={best_alpha} "
            f"r2={model_r2:.6f} mae={model_mae:.6f}"
        )
    else:
        for idx, row in df.iterrows():
            if row["status"] == "OK":
                df.at[idx, "status"] = "学習失敗"
                df.at[idx, "error"] = f"TrainN不足({train_n} < {MIN_TRAIN_N})"

        df["predicted_market_cap"] = np.nan
        df["residual_log"] = np.nan
        df["mispricing_ratio"] = np.nan
        df["overall_rank"] = np.nan
        df["sector_rank"] = np.nan

        log(f"WARN train skipped train_n={train_n} min_required={MIN_TRAIN_N}")

    df["model_r2"] = model_r2
    df["model_mae"] = model_mae
    df["train_n"] = train_n

    output_rows = dataframe_to_sheet_rows(df)
    start_row = 2
    end_row = start_row + len(output_rows) - 1
    write_range = f"D{start_row}:S{end_row}"

    cache_rows = cache_to_sheet_rows(enriched_rows)
    cache_end_col = CACHE_START_COL + len(CACHE_HEADERS) - 1
    cache_header_range = f"{rowcol_to_a1(1, CACHE_START_COL)}:{rowcol_to_a1(1, cache_end_col)}"
    cache_write_range = f"{rowcol_to_a1(start_row, CACHE_START_COL)}:{rowcol_to_a1(end_row, cache_end_col)}"

    log(f"INFO sheet write range={write_range} rows={len(output_rows)}")
    ws.batch_clear(["D2:S"])
    ws.update(values=output_rows, range_name=write_range, value_input_option="USER_ENTERED")
    ws.update(values=[CACHE_HEADERS], range_name=cache_header_range, value_input_option="USER_ENTERED")
    ws.update(values=cache_rows, range_name=cache_write_range, value_input_option="USER_ENTERED")

    status_summary = summarize_status_counts(enriched_rows if train_n >= MIN_TRAIN_N else df.to_dict("records"))
    if pd.isna(model_r2):
        r2_text = "NA"
    else:
        r2_text = f"{model_r2:.6f}"

    if pd.isna(model_mae):
        mae_text = "NA"
    else:
        mae_text = f"{model_mae:.6f}"

    log(f"INFO done {status_summary}, train_n={train_n}, r2={r2_text}, mae={mae_text}")


if __name__ == "__main__":
    main()
