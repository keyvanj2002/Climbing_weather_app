# Australian Weather Assistant 🌤️

Streamlit chatbot powered by LangChain, Groq, and Australian BOM forecast data. Supports daily and filtered hourly forecasts.

## Run locally

Requires Python 3.10+ (3.11 recommended).

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set your **new/current** `GROQ_API_KEY` there. Never commit `.env`.

```bash
streamlit run app.py
```

The original CLI is also supported:

```bash
python weather_agent.py "What's the weather in Horsham tomorrow?"
```

## Deploy free on Streamlit Community Cloud

1. Push `app.py`, `weather_agent.py`, `requirements.txt`, `.gitignore`, and `README.md` to a GitHub repository. **Do not push `.env` or secrets.**
2. Visit https://share.streamlit.io and select **Create app**. Choose your GitHub repository, branch, and `app.py` as the entrypoint.
3. In the app's **Advanced settings / Secrets** (or the deployed app's Settings → Secrets), enter:

   ```toml
   GROQ_API_KEY = "your_actual_key"
   ```

4. Deploy and test a question such as “How windy will Horsham be tomorrow afternoon?”

Free hosting and Groq free-tier usage have limits; a public app can consume your Groq quota. Consider rate limiting or access restrictions before sharing broadly.

## Notes

- Uses `Australia/Melbourne` timezone for hourly forecasts; adjust for other Australian states.
- The geocoder selects the first matching Australian place name; ambiguous towns may need improved disambiguation.
- The chat retains up to six recent user/assistant exchanges as model context, while displaying all messages in the current Streamlit session.
- Current code uses blocking `requests.get` in the async tools; adequate for a small prototype, not ideal for high traffic.
- BOM forecasts are not real-time observations. API and package compatibility can change; pin tested dependency versions for production.
