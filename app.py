from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from emotion_model import EMOTION_LABELS, dataset_summary, get_available_languages, load_language_split, predict_emotions, sample_rows

st.set_page_config(page_title="Emotion Detection Demo", page_icon="😊", layout="wide")


@st.cache_data
def load_summary():
    return dataset_summary()


summary = load_summary()

st.title("SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection")
st.caption("A lightweight app built from the multilingual emotion dataset used in the task.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Languages", summary["language_count"])
col2.metric("Examples", summary["total_examples"])
col3.metric("Emotion labels", len(summary["labels"]))
col4.metric("Tracks", "A / B / C")

st.markdown("---")

left, right = st.columns([1.2, 1.8])
with left:
    languages = summary["languages"]
    language = st.selectbox("Choose a language", languages, index=languages.index("eng") if "eng" in languages else 0)
    df = load_language_split(language, split="train")
    st.write(f"Loaded {len(df)} examples for **{language}**.")

    if not df.empty:
        label_totals = df[EMOTION_LABELS].sum().sort_values(ascending=False)
        st.bar_chart(label_totals)

with right:
    st.subheader("Try the emotion detector")
    user_text = st.text_area(
        "Enter a text snippet",
        value="I am terrified and furious after the attack.",
        height=150,
    )
    if st.button("Analyze emotion"):
        prediction = predict_emotions(user_text)
        emotion_scores = pd.Series(prediction).sort_values(ascending=False)

        st.write("Predicted emotion scores:")
        for label, score in emotion_scores.items():
            st.progress(min(1.0, max(0.0, float(score))))
            st.write(f"{label.title()}: {score:.2f}")

        top_label = emotion_scores.index[0]
        st.success(f"Most likely emotion: **{top_label.title()}**")

    st.subheader("Dataset examples")
    examples = sample_rows(language, n=5)
    if not examples.empty:
        st.dataframe(examples[["text", *EMOTION_LABELS]].head(5), use_container_width=True)

st.markdown("---")
st.subheader("About the task")
st.markdown(
    "This app demonstrates how a dataset-driven emotion model can classify short texts into labels such as anger, fear, joy, sadness, and surprise. The labels are aligned with the SemEval 2025 Task 11 benchmark and can be extended to multilingual or intensity-based settings."
)
