import logging

import streamlit as st

from app.bootstrap import ensure_src_on_path

ensure_src_on_path()

from hse_text_analyzer.application.ports.document_parser import UploadedDocument  # noqa: E402
from hse_text_analyzer.application.services.composition import build_analysis_workflow  # noqa: E402
from hse_text_analyzer.configuration import get_settings  # noqa: E402
from hse_text_analyzer.domain.enums import DocumentType, AmbiguityCategory  # noqa: E402
from hse_text_analyzer.domain.exceptions import HSETextAnalyzerError  # noqa: E402

logger = logging.getLogger(__name__)
settings = get_settings()

st.set_page_config(page_title="Анализ документа", layout="wide")

st.title("Анализ документа")
st.write(
    "Загрузите учебный или академический текст. "
    "На этом этапе приложение выполняет разбор файла."
)

document_type_label_to_value = {
    "Не указан": DocumentType.OTHER,
    "Задание": DocumentType.ASSIGNMENT,
    "Лабораторная работа": DocumentType.LAB_WORK,
    "Отчет": DocumentType.REPORT,
    "Эссе": DocumentType.ESSAY,
    "Проектная документация": DocumentType.PROJECT_DOCUMENTATION,
    "Требования": DocumentType.REQUIREMENTS_SPECIFICATION,
}

with st.sidebar:
    st.header("Параметры")
    selected_document_type = st.selectbox(
        "Тип документа",
        options=list(document_type_label_to_value.keys()),
        index=0,
    )

uploaded_file = st.file_uploader(
    "Документ",
    type=["txt", "md", "docx", "pdf"],
    accept_multiple_files=False,
)

if uploaded_file is None:
    st.info("Выберите файл, чтобы запустить Stage 2 pipeline.")
    st.stop()

if uploaded_file.size > settings.max_upload_size_mb * 1024 * 1024:
    st.error(f"Файл больше допустимого размера: {settings.max_upload_size_mb} МБ.")
    st.stop()

if st.button("Разобрать документ", type="primary"):
    try:
        workflow = build_analysis_workflow()
        uploaded_document = UploadedDocument(
            file_name=uploaded_file.name,
            content=uploaded_file.getvalue(),
            declared_document_type=document_type_label_to_value[selected_document_type],
        )
        with st.spinner("Разбираю документ..."):
            document, result = workflow.run(uploaded_document)

        st.session_state["last_document"] = document
        st.session_state["last_result"] = result
        st.success("Документ успешно разобран.")
    except HSETextAnalyzerError as exc:
        logger.exception("Application error while analyzing document")
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error while analyzing document")
        st.error(f"Не удалось обработать документ: {exc}")

document = st.session_state.get("last_document")
result = st.session_state.get("last_result")

if document and result:
    if document.metadata.warnings:
        for warning in document.metadata.warnings:
            st.warning(warning)

    cols = st.columns(5)
    cols[0].metric("Символов", document.character_count)
    cols[1].metric("Слов", document.word_count)
    if document.linguistic_analysis is not None:
        cols[2].metric("Предложений", document.linguistic_analysis.sentence_count)
        cols[3].metric("Токенов", document.linguistic_analysis.token_count)
    cols[4].metric("Неоднозначностей", result.summary.total_findings)

    if document.linguistic_analysis is not None:
        st.subheader("Результаты лингвистического анализа")
        sentences = document.linguistic_analysis.sentences
        sentence_options = [
            f"Предложение {idx + 1}: {sent.text[:80]}..." if len(sent.text) > 80 else f"Предложение {idx + 1}: {sent.text}"
            for idx, sent in enumerate(sentences)
        ]
        
        if sentence_options:
            selected_sent_str = st.selectbox("Выберите предложение для детального анализа", sentence_options)
            selected_sent_idx = sentence_options.index(selected_sent_str)
            selected_sent = sentences[selected_sent_idx]
            
            st.markdown(f"**Полный текст предложения:** *{selected_sent.text}*")
            
            token_data = []
            for t_idx, t in enumerate(selected_sent.tokens):
                parent_text = "ROOT"
                if t.syntax_parent_idx is not None and 0 <= t.syntax_parent_idx < len(selected_sent.tokens):
                    parent_text = selected_sent.tokens[t.syntax_parent_idx].text
                
                morph_info = []
                m = t.morphology
                if m.pos: morph_info.append(f"POS: {m.pos}")
                if m.lemma: morph_info.append(f"Лемма: {m.lemma}")
                if m.gender: morph_info.append(f"Род: {m.gender}")
                if m.number: morph_info.append(f"Число: {m.number}")
                if m.case: morph_info.append(f"Падеж: {m.case}")
                if m.tense: morph_info.append(f"Время: {m.tense}")
                if m.aspect: morph_info.append(f"Вид: {m.aspect}")
                if m.person: morph_info.append(f"Лицо: {m.person}")
                if m.voice: morph_info.append(f"Залог: {m.voice}")
                
                token_data.append({
                    "№": t_idx + 1,
                    "Токен": t.text,
                    "Часть речи": m.pos or ("Пункт." if t.is_punctuation else ""),
                    "Лемма": m.lemma or "",
                    "Морфологические признаки": ", ".join(morph_info) if morph_info else "Нет данных",
                    "Синтаксический родитель": f"«{parent_text}» (№{t.syntax_parent_idx + 1})" if t.syntax_parent_idx is not None else "ROOT",
                    "Синтаксическая связь": t.syntax_relation or "",
                })
            
            st.dataframe(token_data, width="stretch", hide_index=True)

            def generate_dot_graph(tokens, override=None) -> str:
                dot = [
                    "digraph G {",
                    "  rankdir=TB;",
                    '  node [shape=box, style="filled,rounded", color="#CBD5E1", fillcolor="#F8FAFC", fontname="Arial", fontsize=10];',
                    '  edge [color="#64748B", fontname="Arial", fontsize=8];',
                    '  bgcolor="transparent";'
                ]
                
                override_token_idx = None
                override_parent_idx = None
                override_rel = None
                if override:
                    override_token_idx = override.get("token_idx")
                    override_parent_idx = override.get("parent_idx")
                    override_rel = override.get("relation")

                for idx, t in enumerate(tokens):
                    pos = t.morphology.pos or ("PUNCT" if t.is_punctuation else "UNKNOWN")
                    label = f"#{idx + 1}: {t.text}\\n({pos})"
                    
                    if override_token_idx is not None and idx == override_token_idx:
                        dot.append(f'  node_{idx} [label="{label}", fillcolor="#FEE2E2", color="#EF4444", penwidth=2];')
                    else:
                        dot.append(f'  node_{idx} [label="{label}"];')

                for idx, t in enumerate(tokens):
                    parent_idx = t.syntax_parent_idx
                    rel = t.syntax_relation or ""
                    
                    if override_token_idx is not None and idx == override_token_idx:
                        parent_idx = override_parent_idx
                        rel = override_rel or ""
                        
                    if parent_idx is not None and 0 <= parent_idx < len(tokens):
                        if override_token_idx is not None and idx == override_token_idx:
                            dot.append(f'  node_{parent_idx} -> node_{idx} [label=" {rel}", color="#EF4444", penwidth=2];')
                        else:
                            dot.append(f'  node_{parent_idx} -> node_{idx} [label=" {rel}"];')
                            
                dot.append("}")
                return "\n".join(dot)

            sentence_findings = [
                f for f in result.findings
                if f.span.start_char >= selected_sent.start_char and f.span.end_char <= selected_sent.end_char
            ]

            if sentence_findings:
                st.subheader("Выявленные неоднозначности в предложении")
                for f in sentence_findings:
                    category_ru = {
                        AmbiguityCategory.SYNTACTIC_AMBIGUITY: "Синтаксическая неоднозначность",
                        AmbiguityCategory.REFERENTIAL_AMBIGUITY: "Референциальная неоднозначность",
                        AmbiguityCategory.COORDINATIONAL_AMBIGUITY: "Координационная неоднозначность",
                    }.get(f.category, f.category.value)
                    
                    st.info(
                        f"**[{category_ru}] {f.title}**\n\n"
                        f"**Неоднозначный фрагмент**: {f.span.text}\n\n"
                        f"**Объяснение**: {f.description}\n\n"
                        f"**Вариант исправления**: {f.suggested_revision or 'Не предложен'}"
                    )

            st.markdown("### Синтаксическая структура (AST)")
            
            syntax_finding = None
            for f in sentence_findings:
                if f.category == AmbiguityCategory.SYNTACTIC_AMBIGUITY:
                    if f.evidence and f.evidence[0].features:
                        syntax_finding = f
                        break
            
            if syntax_finding:
                feat = syntax_finding.evidence[0].features
                amb_idx = feat.get("ambiguous_token_idx")
                alt_parent = feat.get("alternative_parent_idx")
                alt_rel = feat.get("alternative_relation", "")
                
                if amb_idx is not None and 1 <= amb_idx <= len(selected_sent.tokens):
                    token_0based = int(amb_idx) - 1
                    parent_0based = int(alt_parent) - 1 if alt_parent is not None and alt_parent > 0 else None
                    
                    override = {
                        "token_idx": token_0based,
                        "parent_idx": parent_0based,
                        "relation": alt_rel,
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Исходный разбор (Natasha)**")
                        st.graphviz_chart(generate_dot_graph(selected_sent.tokens))
                    with col2:
                        st.markdown("**Альтернативный разбор (ИИ)**")
                        st.graphviz_chart(generate_dot_graph(selected_sent.tokens, override=override))
                else:
                    st.graphviz_chart(generate_dot_graph(selected_sent.tokens))
            else:
                st.graphviz_chart(generate_dot_graph(selected_sent.tokens))

    st.subheader("Метаданные")
    st.json(
        {
            "file_name": document.metadata.file_name,
            "file_type": document.metadata.file_type,
            "document_type": document.metadata.document_type,
            "parser": document.metadata.parser_name,
            "sections": len(document.sections),
            "warnings": document.metadata.warnings,
        }
    )

    st.subheader("Предпросмотр текста")
    preview = document.raw_text[:4000] if document.raw_text else "Текст не извлечен."
    st.text_area("Извлеченный текст", value=preview, height=320, disabled=True)

    if result.summary.total_findings == 0:
        st.info("Детекторы неоднозначностей еще не подключены. Это ожидаемо для Stage 2.")
