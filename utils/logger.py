
import os, json, uuid
from datetime import datetime
from pathlib import Path

try:
    import boto3  # Optional
except Exception:
    boto3 = None

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
LOG_DIR = Path(os.getenv("LOG_DIR", DATA_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

AWS_REGION = os.getenv("AWS_REGION")
LOG_S3_BUCKET = os.getenv("LOG_S3_BUCKET", os.getenv("AWS_S3_BUCKET", ""))
LOG_S3_PREFIX = os.getenv("LOG_S3_PREFIX", "logs/")
AWS_OK = bool(AWS_REGION and LOG_S3_BUCKET and boto3)

def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _json_default(o):
    try:
        if hasattr(o, "item"):  # numpy/pandas
            return o.item()
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        return None

def _local_log_path_for_day(day: str | None = None) -> Path:
    day = day or datetime.utcnow().strftime("%Y-%m-%d")
    return LOG_DIR / f"{day}.jsonl"

def _append_jsonl_local(record: dict):
    p = _local_log_path_for_day()
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")

def _put_s3_log_record(record: dict):
    if not AWS_OK:
        return
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)
        ts = record.get("ts") or datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        day = (record.get("ts") or "").split("T")[0] or datetime.utcnow().strftime("%Y-%m-%d")
        rid = record.get("id") or str(uuid.uuid4())
        key = f"{LOG_S3_PREFIX}{day}/{ts}-{rid}.json"
        s3.put_object(Bucket=LOG_S3_BUCKET, Key=key,
                      Body=json.dumps(record, default=_json_default).encode("utf-8"),
                      ContentType="application/json")
    except Exception:
        # best-effort; do not raise
        pass

def _safe_dict(d: dict | None) -> dict:
    d = d or {}
    try:
        json.dumps(d, default=_json_default)
        return d
    except Exception:
        out = {}
        for k, v in d.items():
            try:
                json.dumps(v, default=_json_default)
                out[k] = v
            except Exception:
                out[k] = str(v)
        return out

def log_event(kind: str, league_key: str | None = None, week: int | None = None, data: dict | None = None, ai: dict | None = None):
    record = {
        "id": str(uuid.uuid4()),
        "ts": _utc_now_iso(),
        "kind": kind,
        "league_key": league_key,
        "week": week,
        "data": _safe_dict(data or {}),
        "ai": _safe_dict(ai or {}),
        "app_version": "cli-0.1"
    }
    try:
        _append_jsonl_local(record)
    except Exception:
        pass
    _put_s3_log_record(record)

def read_logs_local(days: int = 30, limit: int = 2000) -> list[dict]:
    rows = []
    today = datetime.utcnow()
    for d in range(days):
        try:
            from pandas import Timedelta
            day = (today - Timedelta(days=d)).strftime("%Y-%m-%d")
        except Exception:
            # fallback without pandas
            day = (today - __import__("datetime").timedelta(days=d)).strftime("%Y-%m-%d")
        p = _local_log_path_for_day(day)
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: 
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("ts",""), reverse=True)
    return rows[:limit]
