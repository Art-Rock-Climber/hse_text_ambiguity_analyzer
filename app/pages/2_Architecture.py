import streamlit as st

from app.bootstrap import ensure_src_on_path

ensure_src_on_path()

st.set_page_config(page_title="Архитектура", layout="wide")

st.title("Архитектура")

st.markdown(
    """
    Приложение разделено на независимые слои:

    - `domain`: чистые модели и доменные исключения;
    - `application`: use cases, ports, registry и aggregation;
    - `infrastructure`: парсеры файлов и LangGraph workflow;
    - `plugins`: будущие анализаторы и ambiguity detectors;
    - `app`: Streamlit UI без бизнес-логики.

    Текущий LangGraph workflow содержит один функциональный узел: загрузка документа,
    запуск зарегистрированных детекторов и агрегация результата. На следующих этапах
    в граф будут добавлены NLP-анализ, rule-based detectors, LLM-анализ и Guardian validation.
    """
)

import os

c4_image_path = os.path.join("docs", "architecture", "c4_architecture.png")
if os.path.exists(c4_image_path):
    st.header("C4 Архитектура")
    st.image(c4_image_path, caption="C4 Architecture Diagram", width="stretch")

sequence_image_path = os.path.join("docs", "architecture", "sequence.png")
if os.path.exists(sequence_image_path):
    st.header("Процесс поиска неоднозначностей")
    st.image(sequence_image_path, caption="Sequence Diagram for Ambiguity Detection", width="stretch")
