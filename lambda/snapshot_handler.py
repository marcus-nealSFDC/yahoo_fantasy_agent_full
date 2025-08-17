
import os, json
from pathlib import Path
from utils.snapshot import run_daily_snapshots

def handler(event, context):
    oauth_path = os.getenv("OAUTH_FILE", "/var/task/oauth2.json")
    output_root = Path(os.getenv("OUTPUT_ROOT", "")) if os.getenv("OUTPUT_ROOT") else None
    try:
        index = run_daily_snapshots(oauth_path=oauth_path, output_root=output_root)
        return {"statusCode": 200, "body": json.dumps(index)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
