"""
Streamlit app — Survey Topic Analyser
Combines Pass 1 (topic discovery) and Pass 2 (classification) in one UI.

Run with:  streamlit run streamlit_app.py
Requires:  pip install anthropic streamlit pandas
export ANTHROPIC_API_KEY=sk-...
"""

import json
import pandas as pd
import streamlit as st
from dataclasses import asdict
import io
import os

from pass1_topic_discovery import discover_topics
from pass2_classification import classify_all

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Автокодировка", layout="wide")
st.title("Автокодировка открытых ответов")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "topics": None,
    "results": None,
    "responses": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.header("1 · Загрузите файл с открытыми ответами")

question_context = st.text_input(
    "Формулировка вопроса (по желанию - помогает модели лучше понять контекст)",
    placeholder="например, что именно вам понравилось в рекламе?",
)

uploaded_file = st.file_uploader(
    "Загрузите файл Excel. Формат - только столбцы с открытыми ответами",
    type=["xlsx"],
)

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    st.write(f"Загружено **{len(df_raw)} строк**, **{len(df_raw.columns)} столбцов**.")

    text_col = st.selectbox("Столбцы с открытыми ответами", df_raw.select_dtypes(include=['object', 'string']).columns)
    responses = df_raw[text_col].dropna()
    if "id" in df_raw.columns:
        ids = df_raw.loc[df_raw[text_col].notna(), "id"]
    else:
        ids = [i for i in range(len(responses))]
    
    st.session_state["ids"] = ids
        
    responses = responses[responses.str.len() >= 2].astype(str).tolist()
    st.session_state["responses"] = responses
    st.write(f"Используем **{len(responses)} не пустых ответов**.")

    with st.expander("Посмотреть ответы"):
        st.dataframe(df_raw[[text_col]].head(10))

# ── Step 2: Topic Discovery ───────────────────────────────────────────────────
st.header("2 · Выделение тематик")

col1, col2 = st.columns(2)
sample_size  = col1.slider("Сколько ответов использовать в подвыборке для выделения тематик?", 50, 400, 200)
max_topics   = col2.slider("Максимальное количество тематик", 5, 30, 15)

if st.button("Начать поиск тем", disabled=not st.session_state["responses"]):
    with st.spinner("Анализ ответов с помощью Claude Sonnet…"):
        result = discover_topics(
            responses=st.session_state["responses"],
            sample_size=sample_size,
            max_topics=max_topics,
            question_context=question_context,
        )
    st.session_state["topics"] = result["topics"]
    st.success(f"Найдено {len(result['topics'])} тематик.")

# Show editable topic list
if st.session_state["topics"]:
    st.subheader("Просмотр и редактирование тематик")
    st.caption("Можно скорректировать названия тем или удалить ненужные")

    topics_df = pd.DataFrame(st.session_state["topics"])
    edited = st.data_editor(
        topics_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "id":          st.column_config.TextColumn("ID (snake_case)", width="small"),
            "label":       st.column_config.TextColumn("Label"),
            "description": st.column_config.TextColumn("Description", width="large"),
        },
    )
    # Persist edits
    st.session_state["topics"] = edited.to_dict(orient="records")


# ── Step 3: Classification ────────────────────────────────────────────────────
st.header("3 · Классификация ответов")

batch_size = 20

if st.button(
    "Запустить классификацию",
    disabled=not (st.session_state["topics"] and st.session_state["responses"]),
):
    progress_bar = st.progress(0, text="Starting…")
    total = len(st.session_state["responses"])

    def update_progress(current, total):
        pct = current / total
        progress_bar.progress(pct, text=f"Классифицировано {current} / {total} ответов...")

    with st.spinner("Классификация с помощью Claude Haiku…"):
        results = classify_all(
            raw_texts=st.session_state["responses"],
            topics=st.session_state["topics"],
            batch_size=batch_size,
            progress_callback=update_progress,
        )

    progress_bar.progress(1.0, text="Готово!")
    st.session_state["results"] = results
    st.success(f"Классифицировано {len(results)} ответов.")

# ── Step 4: Results ───────────────────────────────────────────────────────────
if st.session_state["results"]:
    st.header("4 · Результаты")

    results = st.session_state["results"]
    rows = [asdict(r) for r in results]

    # Expand for display: one row per topic assignment
    topic_labels = {t["id"]: t["label"] for t in (st.session_state["topics"] or [])}

    for row in rows:
        row["topics_display"] = ", ".join(
            topic_labels.get(t, t) for t in row["assigned_topics"]
        )
        row["assigned_topics"] = ", ".join(row["assigned_topics"])

    df_results = pd.DataFrame(rows)

    # Filter by topic
    all_topic_labels = ["(all)"] + sorted(topic_labels.values())
    filter_topic = st.selectbox("Filter by topic", all_topic_labels)

    if filter_topic != "(all)":
        filter_id = next((k for k, v in topic_labels.items() if v == filter_topic), None)
        mask = df_results["assigned_topics"].str.contains(filter_id or "", na=False)
        df_display = df_results[mask]
    else:
        df_display = df_results

    st.write(f"Показано **{len(df_display)}** ответов.")
    st.dataframe(
        df_display[["response_id", "original_text", "topics_display", "confidence"]],
        use_container_width=True,
        column_config={
            "response_id":    st.column_config.NumberColumn("ID", width="small"),
            "Текст":  st.column_config.TextColumn("Response", width="large"),
            "Темы": st.column_config.TextColumn("Topics"),
            "Уровень уверенности":     st.column_config.TextColumn("Confidence", width="small")
        },
    )

    # Topic frequency chart
    st.subheader("Частота тематик")
    from collections import Counter
    all_assigned = [
        topic_labels.get(t, t)
        for r in results
        for t in r.assigned_topics
    ]
    freq = pd.DataFrame(Counter(all_assigned).most_common(), columns=["Topic", "Count"])
    st.bar_chart(freq.set_index("Topic"))

    topics_df = pd.DataFrame(st.session_state["topics"])
    ids = pd.DataFrame(st.session_state["ids"])

    binary_topics = pd.DataFrame(columns = topics_df.label)
    for topic in binary_topics.columns:
        binary_topics[topic] = df_results["topics_display"].str.contains(topic).map({True:1, False:0})

    (binary_topics.sum(axis = 1) == 0).map({True:1, False:0})

    binary_topics["Другое / Затруднились"] = (binary_topics.sum(axis = 1) == 0).map({True:1, False:0})

    df_results = pd.concat([ids, df_results, binary_topics], axis = 1)

    # Download results
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, sheet_name='Кодировка', index=False)
        topics_df.to_excel(writer, sheet_name='Кодфрейм')
        writer.close()

    st.download_button(
        "Скачать результаты",
        data=buffer,
        file_name="autocoding.xlsx",
        mime="application/vnd.ms-excel")
