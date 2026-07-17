"""Streamlit GUI for tell -- record or upload audio, transcribe it locally
with Whisper, extract the same documented text+audio features the CLI uses,
and show the full arithmetic behind the model's output instead of hiding it
behind a single opaque score.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tell.features import audio as audio_features
from tell.features import text as text_features
from tell.model import explain, load_model

MODEL_PATH = Path(__file__).parent / "models" / "tell_classifier.skops"
METADATA_PATH = Path(__file__).parent / "models" / "metadata.json"

st.set_page_config(page_title="tell -- experimental deception-cue explorer", page_icon="\U0001f9ea")


def load_metadata() -> dict | None:
    import json

    if not METADATA_PATH.exists():
        return None
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def render_disclaimer() -> None:
    st.error(
        "**This is an experimental research artifact, not a validated lie detector.** "
        "No published cue -- linguistic, vocal, or otherwise -- has been shown to reliably "
        "detect deception in an individual case. This tool exists to honestly test a research "
        "question and show its real, weak, above-chance-but-far-from-diagnostic result. "
        "**Do not use this to make a real decision about a real person.** Treat every number "
        "below as a fun, honest 'what if' about a future research direction -- not a verdict.",
        icon="⚠️",
    )


def get_audio_bytes() -> bytes | None:
    st.subheader("1. Provide a recording")
    tab_record, tab_upload = st.tabs(["Record with your microphone", "Upload a file"])
    with tab_record:
        recorded = st.audio_input("Speak, then stop the recording")
        if recorded is not None:
            return recorded.getvalue()
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a .wav/.mp3/.m4a file", type=["wav", "mp3", "m4a", "ogg"]
        )
        if uploaded is not None:
            return uploaded.getvalue()
    return None


def main() -> None:
    st.title("tell")
    st.caption("An honest experiment in combining documented (and weak) deception cues.")
    render_disclaimer()

    metadata = load_metadata()
    model = get_model()

    if model is None:
        st.warning(
            "No trained model found at models/tell_classifier.skops. Run "
            "`python scripts/train_and_save_model.py` first."
        )
        return

    if metadata:
        cols = st.columns(4)
        acc_mean = metadata["cross_validated_accuracy_mean"]
        acc_std = metadata["cross_validated_accuracy_std"]
        cols[0].metric("Model's own cross-validated accuracy", f"{acc_mean:.1%} ± {acc_std:.1%}")
        cols[1].metric("Precision", f"{metadata['precision']:.1%}")
        cols[2].metric("Recall", f"{metadata['recall']:.1%}")
        cols[3].metric("Trained on", f"n={metadata['n_examples']} real clips")
        st.caption(
            "Chance is 50%. Keep this number in view next to every score below -- "
            "it is the honest ceiling on how much any single prediction here should be trusted."
        )

    audio_bytes = get_audio_bytes()
    if audio_bytes is None:
        st.info("Record or upload audio above to continue.")
        return

    st.audio(audio_bytes)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    st.subheader("2. Transcript")
    if "transcript" not in st.session_state:
        st.session_state["transcript"] = ""

    if st.button("Transcribe with local Whisper model"):
        try:
            from tell.transcribe import transcribe

            with st.spinner("Transcribing locally (first run downloads the Whisper model)..."):
                st.session_state["transcript"] = transcribe(audio_path)
        except Exception as exc:  # noqa: BLE001 -- surfaced directly to the user, not swallowed
            st.error(f"Transcription failed: {exc}")

    transcript = st.text_area(
        "Transcript (edit freely -- Whisper isn't perfect, and text features depend on it)",
        value=st.session_state["transcript"],
        height=100,
    )

    if not transcript.strip():
        st.info("Provide a transcript (transcribe or type it) to run the analysis.")
        return

    st.subheader("3. Extracted features")
    text_feats = text_features.extract_text_features(transcript)
    audio_feats = audio_features.extract_audio_features(audio_path)
    feature_vector = text_features.features_to_vector(
        text_feats
    ) + audio_features.features_to_vector(audio_feats)

    col_text, col_audio = st.columns(2)
    with col_text:
        st.markdown("**Text features**")
        st.dataframe(pd.DataFrame([text_feats.__dict__]).T.rename(columns={0: "value"}))
    with col_audio:
        st.markdown("**Audio features**")
        st.dataframe(pd.DataFrame([audio_feats.__dict__]).T.rename(columns={0: "value"}))

    st.subheader("4. The math, in full")
    contributions = explain(model, feature_vector)
    intercept = float(model.named_steps["clf"].intercept_[0])

    df = pd.DataFrame([c.__dict__ for c in contributions]).sort_values(
        "contribution_to_logit", key=abs, ascending=True
    )
    fig = go.Figure(
        go.Bar(
            x=df["contribution_to_logit"],
            y=df["name"],
            orientation="h",
            marker_color=["#c0392b" if v > 0 else "#2980b9" for v in df["contribution_to_logit"]],
        )
    )
    fig.update_layout(
        title="Per-feature contribution to the logit (z-score x coefficient)",
        xaxis_title="contribution (positive = pushes toward 'deceptive')",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    logit = intercept + float(sum(c.contribution_to_logit for c in contributions))
    probability = sigmoid(logit)

    contribution_sum = logit - intercept
    st.markdown(
        f"""
**Step by step:**
1. Standardize each feature: `z = (value - training_mean) / training_std`
2. Multiply by the trained coefficient: `contribution = z * coefficient` (chart above)
3. Sum every contribution plus the intercept:
   `logit = {intercept:.3f} + {contribution_sum:.3f} = {logit:.3f}`
4. Pass through the logistic sigmoid: `probability = 1 / (1 + e^-logit) = {probability:.1%}`
"""
    )

    accuracy_phrase = (
        f"{metadata['cross_validated_accuracy_mean']:.0%}" if metadata else "about two-thirds"
    )

    st.subheader("5. Result")
    st.metric(
        "Model classification signal (NOT a probability this person is lying)", f"{probability:.1%}"
    )
    st.caption(
        "This number is exactly step 4 above -- nothing more. It reflects how this specific "
        "model's coefficients, fit on 121 real but imperfectly-labeled clips, react to these "
        f"specific standardized feature values. Given the model's own cross-validated accuracy "
        f"is roughly {accuracy_phrase} on held-out data -- barely better than a coin flip with "
        "real uncertainty -- this is a curiosity about a research direction, not a verdict "
        "about you or anyone else."
    )

    with st.expander(
        "Optional: fact-check specific claims in this transcript (separate, unrelated signal)"
    ):
        st.caption(
            "This checks whether specific factual claims are supported by web search evidence "
            "-- a completely different question from the behavioral cues above, kept separate "
            "on purpose."
        )
        api_key = st.text_input(
            "Anthropic API key (used only for this request, never stored)", type="password"
        )
        if st.button("Check facts") and api_key:
            try:
                from tell.factcheck import get_factcheck_provider

                provider = get_factcheck_provider("anthropic", api_key=api_key)
                with st.spinner("Extracting and checking claims..."):
                    verdicts = provider.verify_transcript(transcript)
                if not verdicts:
                    st.info(
                        "No independently checkable factual claims were found in this transcript."
                    )
                for v in verdicts:
                    st.write(f"**{v.claim}** -- {v.verdict}")
                    st.caption(v.explanation)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Fact-check failed: {exc}")


if __name__ == "__main__":
    main()
