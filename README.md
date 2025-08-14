# Yahoo Fantasy Agent — Live (Roster + Start/Sit + Waivers)

A minimal but full-featured Streamlit app that connects to Yahoo Fantasy Sports (OAuth),
shows your leagues & roster, recommends a lineup (heuristic), and submits add/drop or FAAB waiver claims.

## 1) Install
```bash
pip3 install -r requirements.txt
```

## 2) Configure
Create a `.env` next to this file (see `.env.example`) and set:
- `DEV_MODE=0`
- `YAHOO_CLIENT_ID`
- `YAHOO_CLIENT_SECRET`
- `YAHOO_REDIRECT_URI`  (MUST exactly match your Yahoo app's Redirect URI)
- `OPENAI_API_KEY` (optional; enables "Explain" rationale)

Yahoo will reject mismatches. For Streamlit Cloud, use your app's `https://<app>.streamlit.app` URL (no trailing slash).
For local dev, `http://localhost:8501` is OK (set the Yahoo app Redirect to that while testing).

## 3) Run
```bash
streamlit run streamlit_app.py
```
Click **Connect Yahoo**, approve, and you'll see your leagues and roster.

## Notes
- Tokens are saved to `oauth2.json` (ignored by git).
- Waiver execution tries immediate add/drop first, then known waiver methods if needed.
- OPENAI rationale is optional and wrapped in try/except so it won't block if not configured.
