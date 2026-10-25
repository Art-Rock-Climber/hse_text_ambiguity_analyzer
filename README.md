# HSE Text Analyzer

Streamlit application for detecting ambiguities in Russian academic texts.

## What it does

- Uploads `.txt`, `.md`, `.docx`, `.pdf` files.
- Parses documents into a clean domain model.
- Segments Russian text into sentences and tokens using **Natasha**.
- Performs morphological and syntactic analysis using **pymorphy3** and the Natasha dependency parser.
- Runs three AI-based ambiguity detectors (syntactic, referential, coordinational) with morphological prefiltering to minimise unnecessary API calls.
- Supports two AI providers: **OpenRouter** (default) and **Groq** — switchable via the `ACTIVE_AI_PROVIDER` secret without code changes.
- Renders detected ambiguities and side-by-side AST dependency trees in the Streamlit UI.

## Run Locally

```bash
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

Create `.streamlit/secrets.toml` with your API credentials:

```toml
OPENROUTER_API_KEY = "sk-or-..."
OPENROUTER_MODEL = "google/gemini-2.5-flash"
OPENROUTER_MODEL_EXTRA_1 = "mistralai/mistral-7b-instruct"   # fallback on 429

# Optional — switch to Groq:
# ACTIVE_AI_PROVIDER = "groq"
# GROQ_API_KEY = "gsk_..."
# GROQ_MODEL = "llama3-70b-8192"
```

## Supported Uploads

| Format | Parser |
|---|---|
| `.txt` | `BasePlainTextParser` |
| `.md` | `BasePlainTextParser` |
| `.docx` | `python-docx` |
| `.pdf` | `PyMuPDF` |

## Run Tests

```bash
python -m pytest
```

All 13 unit tests run without API credentials (AI methods are globally mocked via `conftest.py`).

## Architecture

See the **Architecture** page in the running app, or the PlantUML source in `docs/architecture/`.
