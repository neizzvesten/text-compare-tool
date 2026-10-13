from __future__ import annotations
import io
import html
import pandas as pd
import streamlit as st
from text_compare_tool import Analyzer, load_file, save_html_report

st.set_page_config(page_title="Сравнение текстов", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem;}
.main-title {font-size: 2.25rem; font-weight: 750; letter-spacing: -0.03em;}
.subtitle {color: #64748b; font-size: 1rem; margin-bottom: 1.5rem;}
div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0; padding: 14px 16px; border-radius: 12px;}
.match-card {border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin: 10px 0 14px 0; background: white;}
.match-type {font-size: 0.85rem; font-weight: 700; color: #334155; margin-bottom: 8px;}
.match-text {background: #f8fafc; padding: 12px; border-radius: 8px; white-space: pre-wrap; overflow-wrap: anywhere;}
.disclaimer {font-size: .9rem; color: #475569; background: #f8fafc; padding: 14px; border-radius: 10px; border: 1px solid #e2e8f0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Сравнение двух текстов</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Прототип для первичного поиска и предварительной классификации текстовых совпадений</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Параметры анализа")
    min_words = st.slider("Минимальная длина дословного совпадения (слов)", 3, 12, 5)
    partial_threshold = st.slider("Порог частичного совпадения", 0.50, 0.95, 0.65, 0.05)
    lexical_threshold = st.slider("Порог лексического сходства", 0.25, 0.90, 0.50, 0.05)
    grammatical_threshold = st.slider("Порог грамматического сходства", 0.50, 0.95, 0.72, 0.05)
    semantic_threshold = st.slider("Порог лексико-семантического сходства", 0.30, 0.95, 0.62, 0.05)
    st.caption("Пороги предварительные. Для экспертного применения их необходимо проверить на размеченном корпусе.")

@st.cache_resource
def get_analyzer():
    return Analyzer()

@st.cache_data(show_spinner=False)
def read_uploaded_file(name: str, data: bytes) -> str:
    # load_file expects a path, so decode TXT directly and use temporary files for DOCX/PDF.
    import tempfile, os
    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        temp_path = f.name
    try:
        return load_file(temp_path)
    finally:
        try: os.unlink(temp_path)
        except OSError: pass


def load_controls(label: str, key: str):
    uploaded = st.file_uploader(label, type=["txt", "md", "docx", "pdf"], key=key)
    if uploaded:
        try:
            return read_uploaded_file(uploaded.name, uploaded.getvalue())
        except Exception as e:
            st.error(f"Не удалось прочитать файл: {e}")
    return ""

with st.expander("Как пользоваться", expanded=False):
    st.write("Вставьте тексты в два поля или загрузите файлы TXT, DOCX или PDF. Нажмите «Сравнить тексты». Результаты показывают потенциальные совпадения и требуют экспертной проверки.")

left_upload, right_upload = st.columns(2)
with left_upload:
    uploaded_a = load_controls("Загрузить текст 1 (необязательно)", "file_a")
with right_upload:
    uploaded_b = load_controls("Загрузить текст 2 (необязательно)", "file_b")

example_a = "В ходе исследования были получены следующие результаты. Компания понесла значительные финансовые потери. Учёные провели исследование влияния стресса."
example_b = "В ходе исследования были получены следующие результаты. Компания понесла существенные финансовые потери. Исследование влияния стресса было проведено учёными."

# Update text-area state before the widgets are instantiated.
if uploaded_a and st.session_state.get("_last_upload_a") != uploaded_a:
    st.session_state["text_a"] = uploaded_a
    st.session_state["_last_upload_a"] = uploaded_a
if uploaded_b and st.session_state.get("_last_upload_b") != uploaded_b:
    st.session_state["text_b"] = uploaded_b
    st.session_state["_last_upload_b"] = uploaded_b

if st.button("Загрузить демонстрационный пример"):
    st.session_state["text_a"] = example_a
    st.session_state["text_b"] = example_b

col_a, col_b = st.columns(2, gap="large")
with col_a:
    st.subheader("Текст 1")
    text_a = st.text_area("Вставьте первый текст", value=uploaded_a, height=260, key="text_a", label_visibility="collapsed", placeholder="Вставьте текст или загрузите файл выше…")
with col_b:
    st.subheader("Текст 2")
    text_b = st.text_area("Вставьте второй текст", value=uploaded_b, height=260, key="text_b", label_visibility="collapsed", placeholder="Вставьте текст или загрузите файл выше…")

compare = st.button("🔎 Сравнить тексты", type="primary", use_container_width=True)

if compare:
    if not text_a.strip() or not text_b.strip():
        st.warning("Добавьте оба текста перед сравнением.")
    else:
        with st.spinner("Сравниваем тексты… Для больших документов анализ может занять некоторое время."):
            analyzer = get_analyzer()
            # Apply selected thresholds explicitly so controls change the actual analysis.
            a = analyzer.normalize_text(text_a)
            b = analyzer.normalize_text(text_b)
            exact = analyzer.exact_matches(a, b, min_words=min_words)
            partial = analyzer.partial_matches(a, b, low=partial_threshold)
            lexical = analyzer.lexical_matches(a, b, threshold=lexical_threshold)
            grammatical = analyzer.grammatical_matches(a, b, threshold=grammatical_threshold)
            semantic, semantic_method = analyzer.semantic_matches(a, b, threshold=semantic_threshold)
            groups = [exact, partial, lexical, grammatical, semantic]
            matches, seen = [], set()
            for group in groups:
                for m in group:
                    key = (m.category, analyzer.normalize_text(m.text1).lower(), analyzer.normalize_text(m.text2).lower())
                    if key not in seen:
                        seen.add(key)
                        from dataclasses import asdict
                        matches.append(asdict(m))
            counts = {
                "Полнотекстовые / дословные": len(exact),
                "Частичные": len(partial),
                "Лексические": len(lexical),
                "Грамматические / синтаксические": len(grammatical),
                "Семантические / лексико-статистические": len(semantic),
            }
            st.session_state["analysis_result"] = {"counts": counts, "matches": matches, "semantic_method": semantic_method,
                "notes": ["Результаты являются поисковой подсказкой и не заменяют лингвистическую интерпретацию.",
                "Дословные совпадения ищутся по последовательностям слов; пунктуация может не учитываться.",
                "Если использован TF-IDF, он выявляет преимущественно лексическое сходство и не обеспечивает полноценного понимания смысла."]}

result = st.session_state.get("analysis_result")
if result:
    st.divider()
    st.header("Результаты сравнения")
    total = sum(result["counts"].values())
    st.caption(f"Найдено совпадений/кандидатов: {total}. Один и тот же фрагмент может отражаться в нескольких категориях.")
    metric_cols = st.columns(5)
    for col, (label, count) in zip(metric_cols, result["counts"].items()):
        short_label = {"Полнотекстовые / дословные":"Дословные", "Грамматические / синтаксические":"Грамматические", "Семантические / лексико-статистические":"Семантические*"}.get(label, label)
        col.metric(short_label, count)
    st.caption("* Семантическая оценка зависит от доступного метода; смотрите пояснение ниже.")

    if result["matches"]:
        df = pd.DataFrame(result["matches"])
        df["score_percent"] = (df["score"] * 100).round(1)
        st.subheader("Список найденных фрагментов")
        categories = sorted(df["category"].unique().tolist())
        selected = st.multiselect("Фильтр по типу совпадения", categories, default=categories)
        view = df[df["category"].isin(selected)]
        st.dataframe(view[["category", "score_percent", "text1", "text2", "details"]].rename(columns={"category":"Тип", "score_percent":"Сходство, %", "text1":"Фрагмент текста 1", "text2":"Фрагмент текста 2", "details":"Комментарий"}), use_container_width=True, hide_index=True)
        st.subheader("Фрагменты подробно")
        for i, m in enumerate(view.to_dict("records"), start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {m['category']}** · сходство: **{m['score']:.0%}**")
                c1, c2 = st.columns(2)
                c1.markdown("**Текст 1**")
                c1.write(m["text1"])
                c2.markdown("**Текст 2**")
                c2.write(m["text2"])
                st.caption(m.get("details", ""))
        csv = view[["category", "score", "text1", "text2", "details"]].to_csv(index=False).encode("utf-8-sig")
        st.download_button("Скачать результаты CSV", data=csv, file_name="text_comparison_results.csv", mime="text/csv")
        # HTML report generated in memory
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
            report_path = f.name
        try:
            save_html_report(report_path, result)
            with open(report_path, "rb") as f:
                html_report = f.read()
            st.download_button("Скачать HTML-отчёт", data=html_report, file_name="text_comparison_report.html", mime="text/html")
        finally:
            try: os.unlink(report_path)
            except OSError: pass
    else:
        st.info("Совпадения не обнаружены при выбранных порогах. Это не доказывает отсутствие сходства.")
    st.markdown(f"**Метод семантического анализа:** {result['semantic_method']}")
    st.markdown('<div class="disclaimer"><b>Важно:</b> это исследовательский прототип. Автоматическая оценка не устанавливает плагиат и не заменяет экспертную интерпретацию. Перед использованием в экспертной практике необходима проверка на размеченном корпусе.</div>', unsafe_allow_html=True)

st.divider()
st.caption("Конфиденциальность: при запуске локально тексты остаются на вашем компьютере. При публикации приложения через облачный хостинг текст обрабатывается на сервере хостинга; не загружайте материалы ограниченного доступа в публичную облачную версию.")
