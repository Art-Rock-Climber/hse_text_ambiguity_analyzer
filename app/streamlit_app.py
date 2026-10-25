import streamlit as st

from app.bootstrap import ensure_src_on_path

ensure_src_on_path()

from hse_text_analyzer.configuration import get_settings  # noqa: E402
from hse_text_analyzer.shared.logging import configure_logging  # noqa: E402

settings = get_settings()
configure_logging(settings.log_level)

st.set_page_config(
    page_title=settings.app_name,
    layout="wide",
)

st.title(settings.app_name)
st.write(
    "Локальное приложение для анализа неоднозначностей в русскоязычных учебных "
    "и академических текстах."
)

st.subheader("Навигация")
st.page_link("pages/1_Analyze_Document.py", label="Анализ документа")
st.page_link("pages/2_Architecture.py", label="Архитектура")

st.subheader("Что умеет приложение")
st.markdown(
    """
    - загрузка `.txt`, `.md`, `.docx`, `.pdf`;
    - нормализация документа в доменную модель;
    - сегментация русского текста на предложения и токены (Natasha);
    - морфологический и синтаксический анализ токенов (pymorphy3 + Natasha);
    - предфильтрация предложений перед вызовом ИИ;
    - три AI-детектора неоднозначностей (синтаксический, референциальный, координационный);
    - поддержка двух AI-провайдеров: OpenRouter и Groq (переключается через `ACTIVE_AI_PROVIDER`);
    - отображение статистики, неоднозначностей и синтаксических деревьев (AST) в UI.
    """
)

st.subheader("Виды выявляемых неоднозначностей")
st.markdown(
    """
| Вид неоднозначности | Пояснение | Пример |
|:---|:---|:---|
| **Синтаксическая** | фраза имеет несколько разборов по структуре, например привязка уточняющего оборота к разным частям предложения | Я видел человека с биноклем |
| **Координационная** | непонятно, как группируются элементы с «и/или» | Студенты и преподаватели и родители пришли на встречу |
| **Референциальная** | местоимение или указатель могут относиться к разным сущностям | Мария сказала Анне, что она опоздает |
    """
)
