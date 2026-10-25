import streamlit as st
from pathlib import Path

from app.bootstrap import ensure_src_on_path

ensure_src_on_path()

st.set_page_config(page_title="Архитектура", layout="wide")

st.title("Архитектура")

st.markdown(
    """
    Приложение разделено на независимые слои:

    - `domain`: чистые модели и доменные исключения;
    - `application`: use cases, ports, registry и aggregation;
    - `infrastructure`: парсеры файлов, NLP (Natasha + pymorphy3), AI-клиенты и LangGraph workflow;
    - `plugins`: зарегистрированные детекторы неоднозначностей;
    - `app`: Streamlit UI без бизнес-логики.

    LangGraph workflow содержит один функциональный узел: загрузка документа,
    запуск зарегистрированных детекторов и агрегация результата.
    """
)

# Use Path relative to this file so the page works both locally and on Streamlit Cloud.
_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs" / "architecture"

c4_image_path = _DOCS_DIR / "c4_architecture.png"
if c4_image_path.exists():
    st.header("C4 Архитектура")
    st.image(str(c4_image_path), caption="C4 Architecture Diagram", use_container_width=True)

sequence_image_path = _DOCS_DIR / "sequence.png"
if sequence_image_path.exists():
    st.header("Процесс поиска неоднозначностей")
    st.image(str(sequence_image_path), caption="Sequence Diagram for Ambiguity Detection", use_container_width=True)
