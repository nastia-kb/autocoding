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

from pass1_topic_discovery import discover_topics
from pass2_classification import classify_all

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Survey Topic Analyser", layout="wide")
st.title("Survey Topic Analyser")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "topics": None,
    "results": None,
    "responses": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Step 1: Upload ────────────────────────────────────────────────────────────
st.header("1 · Upload responses")

question_context = st.text_input(
    "Survey question (optional — helps the model understand context)",
    placeholder="e.g. What could we improve about your experience?",
)

uploaded_file = st.file_uploader(
    "Upload a CSV file. The column containing responses will be auto-detected,\n"
    "or you can choose it below.",
    type=["csv"],
)

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    st.write(f"Loaded **{len(df_raw)} rows**, **{len(df_raw.columns)} columns**.")

    text_col = st.selectbox("Column containing open-ended responses", df_raw.columns)
    responses = df_raw[text_col].dropna().astype(str).tolist()
    st.session_state["responses"] = responses
    st.write(f"Using **{len(responses)} non-empty responses**.")

    with st.expander("Preview responses"):
        st.dataframe(df_raw[[text_col]].head(10))

# ── Step 2: Topic Discovery ───────────────────────────────────────────────────
st.header("2 · Discover topics (Pass 1)")

col1, col2 = st.columns(2)
sample_size  = col1.slider("Responses to sample for discovery", 10, 200, 50)
max_topics   = col2.slider("Max topics to generate", 5, 20, 12)

if st.button("Run topic discovery", disabled=not st.session_state["responses"]):
    with st.spinner("Analysing responses with Claude Sonnet…"):
        result = discover_topics(
            responses=st.session_state["responses"],
            sample_size=sample_size,
            max_topics=max_topics,
            question_context=question_context,
        )
    st.session_state["topics"] = result["topics"]
    if result.get("model_notes"):
        st.info(f"Model notes: {result['model_notes']}")
    st.success(f"Discovered {len(result['topics'])} topics.")

# Show editable topic list
if st.session_state["topics"]:
    st.subheader("Review & edit topics")
    st.caption("Rename labels or descriptions before running classification. Remove rows to drop a topic.")

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

    # Download topic list
    st.download_button(
        "Download topic list (JSON)",
        data=json.dumps({"topics": st.session_state["topics"]}, indent=2),
        file_name="topics.json",
        mime="application/json",
    )

# ── Step 3: Classification ────────────────────────────────────────────────────
st.header("3 · Classify responses (Pass 2)")

batch_size = st.slider("Responses per API call (higher = fewer calls, same cost)", 5, 20, 10)

if st.button(
    "Run classification",
    disabled=not (st.session_state["topics"] and st.session_state["responses"]),
):
    progress_bar = st.progress(0, text="Starting…")
    total = len(st.session_state["responses"])

    def update_progress(current, total):
        pct = current / total
        progress_bar.progress(pct, text=f"Classified {current} / {total} responses…")

    with st.spinner("Classifying with Claude Haiku…"):
        results = classify_all(
            raw_texts=st.session_state["responses"],
            topics=st.session_state["topics"],
            batch_size=batch_size,
            progress_callback=update_progress,
        )

    progress_bar.progress(1.0, text="Done!")
    st.session_state["results"] = results
    st.success(f"Classified {len(results)} responses.")

# ── Step 4: Results ───────────────────────────────────────────────────────────
if st.session_state["results"]:
    st.header("4 · Results")

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

    st.write(f"Showing **{len(df_display)}** responses.")
    st.dataframe(
        df_display[["response_id", "original_text", "topics_display", "confidence", "summary"]],
        use_container_width=True,
        column_config={
            "response_id":    st.column_config.NumberColumn("ID", width="small"),
            "original_text":  st.column_config.TextColumn("Response", width="large"),
            "topics_display": st.column_config.TextColumn("Topics"),
            "confidence":     st.column_config.TextColumn("Confidence", width="small"),
            "summary":        st.column_config.TextColumn("Summary", width="large"),
        },
    )

    # Topic frequency chart
    st.subheader("Topic frequency")
    from collections import Counter
    all_assigned = [
        topic_labels.get(t, t)
        for r in results
        for t in r.assigned_topics
    ]
    freq = pd.DataFrame(Counter(all_assigned).most_common(), columns=["Topic", "Count"])
    st.bar_chart(freq.set_index("Topic"))

    # Download results
    st.download_button(
        "Download results (CSV)",
        data=df_results.to_csv(index=False),
        file_name="classified_responses.csv",
        mime="text/csv",
    )
