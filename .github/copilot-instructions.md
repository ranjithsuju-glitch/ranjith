# Project Guidance

- This repository is a fictional, synthetic-data capstone; never present its sample records or policies as real Swiggy/Instamart information.
- Use Python 3.11+ and only the standard library, pandas, numpy, scikit-learn, and joblib.
- Keep `generate_data.py` and `mock_llm.py` unmodified. All model calls must use the offline `mock_llm.py`; never add network calls, API keys, or paid services.
- Keep the Parts connected: Part A owns the SQLite database, Part B derives customer features and lapse scores, Part C uses those outputs for ticket routing, and Part D reuses the database and policy retrieval.
- Redact customer identifiers before classification; refund tools must enforce the fictional policy and leave an auditable local ledger.
- Run the numbered Reproduce commands in `README.md` from the repository root.