# HSE Text Analyzer

Streamlit application for detecting ambiguity in Russian academic texts.

Stage 1 provides the foundation:

- Clean Architecture package structure.
- Configuration and logging.
- Common domain models.
- Application ports and services.
- Parser registry and document loading pipeline.
- Plugin registration mechanism for future detectors.
- LangGraph workflow skeleton.
- Streamlit navigation and document upload UI.

Stage 2 adds:

- Sentence and token models for structured linguistic data.
- Russian sentence segmentation and tokenization.
- A morphology adapter based on `pymorphy2`.
- A `LinguisticAnalysis` layer attached to parsed documents.

## Run Locally

```bash
pip install -e ".[dev]"
streamlit run app/streamlit_app.py
```

## Supported Uploads

- `.txt`
- `.md`
- `.docx`
- `.pdf`

PDF and DOCX parsing depend on `pymupdf` and `python-docx`.
