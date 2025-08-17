
#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from utils.snapshot import run_daily_snapshots

def main():
    ap = argparse.ArgumentParser(description="Run daily snapshots + morning briefs for all latest NFL leagues.")
    ap.add_argument("--oauth", default="oauth2.json", help="Path to Yahoo oauth2.json")
    ap.add_argument("--out", default=None, help="Output root (default data/daily/YYYY-MM-DD)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else None
    index = run_daily_snapshots(oauth_path=args.oauth, output_root=out)
    print(json.dumps(index, indent=2))

if __name__ == "__main__":
    main()
