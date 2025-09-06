# app.py
# -----------------------------
# Real-time(ish) Meeting Summarizer — FREE stack
# - Transcribe: faster-whisper (open source)
# - Summarize: DistilBART (free, Hugging Face) with a 2-pass summarization
# - Extract: keyword-based + optional FLAN-T5 (if available) for action items
# - UI: Streamlit
#
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Notes:
# - This runs fully locally and uses only free/open-source models/APIs.
# - For faster-whisper, having ffmpeg installed on your system is recommended.
# - You can switch to lighter models in the sidebar to save RAM/CPU.
# -----------------------------

import os
import io
import math
import tempfile
from typing import List, Dict
import streamlit as st
from transformers import pipeline
from faster_whisper import WhisperModel
import ffmpeg
from pydub import AudioSegment

st.set_page_config(page_title="Meeting Summarizer (Free)", page_icon="📝", layout="wide")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("⚙️ Settings")

asr_model_size = st.sidebar.selectbox(
    "ASR model size (faster-whisper)",
    ["base", "small", "medium"],
    index=0,
    help="Smaller = faster download/inference, lower accuracy. 'medium' is better but heavier."
)

device = st.sidebar.selectbox("Device", ["cpu"], index=0)
compute_type = st.sidebar.selectbox(
    "Compute precision",
    ["int8", "float16", "float32"],
    index=0,
    help="int8 is fastest on CPU; float32 is most accurate but slowest."
)

summ_model_name = st.sidebar.text_input(
    "Summarization model (Hugging Face)",
    value="sshleifer/distilbart-cnn-12-6",
    help="Default: a smaller BART summarizer. You can try 'facebook/bart-large-cnn' if you have more RAM."
)

use_optional_flan = st.sidebar.checkbox(
    "Use optional FLAN-T5 for Action Items (slower, optional)",
    value=False,
    help="If checked, will try 'google/flan-t5-base' (free) for action items and decisions extraction."
)

max_chunk_tokens = st.sidebar.slider(
    "Max words per chunk (approx.)",
    min_value=200, max_value=1200, value=700, step=50,
    help="Transcript is split into chunks for summarization."
)

temperature = st.sidebar.slider(
    "Action item generator creativity (FLAN-T5)",
    min_value=0.0, max_value=1.0, value=0.0, step=0.1
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For long files, start with 'base' + the default summarizer, then scale up.")

# -----------------------------
# Helpers
# -----------------------------

def save_uploaded_file_to_wav(upload) -> str:
    # """Save uploaded audio/video to a temporary .wav file with 16k mono."""
    if upload is None:
        return None
    suffix = os.path.splitext(upload.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        tmp_path = tmp.name

    # If it's an MP3 file, return it directly (faster-whisper can handle MP3)
    if suffix == '.mp3':
        return tmp_path
    
    # Convert to 16k mono wav via pydub (ffmpeg backend)
    audio = AudioSegment.from_file(tmp_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(wav_fd)
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe(wav_path: str, model_size: str, device: str, compute_type: str):
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(wav_path, vad_filter=False)
    text_segments = []
    for seg in segments:
        text_segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip()
        })
    full_text = " ".join(s['text'] for s in text_segments).strip()
    return text_segments, full_text, info

@st.cache_resource(show_spinner=False)
def get_summarizer(model_name: str):
    return pipeline("summarization", model=model_name)

@st.cache_resource(show_spinner=False)
def get_flan():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
        mdl = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return tok, mdl
    except Exception as e:
        return None, None

def chunk_text_by_words(text: str, max_words: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def summarize_long_text(text: str, model_name: str, max_words: int) -> str:
    if not text.strip():
        return ""
    summarizer = get_summarizer(model_name)
    chunks = chunk_text_by_words(text, max_words)
    partial_summaries = []
    for ch in chunks:
        # Conservative lengths to avoid OOM on CPU
        max_len = min(256, max(128, len(ch.split()) // 2))
        min_len = min(120, max(60, len(ch.split()) // 6))
        out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        partial_summaries.append(out)

    # Second pass: summarize the summaries
    joined = " ".join(partial_summaries)
    final = summarizer(joined, max_length=220, min_length=120, do_sample=False)[0]["summary_text"]
    return final

def keyword_extract_items(text: str) -> Dict[str, List[str]]:
    import re
    lines = [ln.strip("-•* ").strip() for ln in text.splitlines() if ln.strip()]
    # simple heuristics for action/decision/risks
    actions, decisions, risks = [], [], []
    patterns_action = [r"we (will|need to|should)\b", r"\bassign(ed)?\b", r"\bdeadline\b", r"\bETA\b", r"\bby (EOD|tomorrow|\d{4}-\d{2}-\d{2})\b"]
    patterns_decision = [r"\bdecided\b", r"\bagreed\b", r"\bapproved\b", r"\bchose\b"]
    patterns_risk = [r"\brisk\b", r"\bblocker\b", r"\bissue\b", r"\bdependency\b"]

    for ln in lines:
        low = ln.lower()
        if any(re.search(p, low) for p in patterns_decision):
            decisions.append(ln)
        elif any(re.search(p, low) for p in patterns_risk):
            risks.append(ln)
        elif any(re.search(p, low) for p in patterns_action):
            actions.append(ln)
    return {"actions": actions, "decisions": decisions, "risks": risks}

def flan_generate_sections(transcript: str, temperature: float = 0.0) -> Dict[str, List[str]]:
    tok, mdl = get_flan()
    if tok is None:
        return {}
    prompt = (
        "You are an assistant that extracts structured meeting minutes.\n"
        "From the transcript below, list:\n"
        "1) Key Decisions (bullets)\n"
        "2) Action Items (bullets with assignee and deadline if present)\n"
        "3) Risks/Blockers (bullets)\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Return in JSON with keys: decisions, actions, risks."
    )
    inputs = tok(prompt, return_tensors="pt")
    gen = mdl.generate(**inputs, max_new_tokens=256, temperature=float(temperature))
    text = tok.decode(gen[0], skip_special_tokens=True)

    # Try to parse JSON; if not, fallback to naive splitting
    import json, re
    try:
        data = json.loads(text)
        for k in ["decisions", "actions", "risks"]:
            if k not in data:
                data[k] = []
        return data
    except Exception:
        # fallback: extract lines after headings
        sections = {"decisions": [], "actions": [], "risks": []}
        cur = None
        for ln in text.splitlines():
            l = ln.strip().lower()
            if "decision" in l:
                cur = "decisions"; continue
            if "action" in l:
                cur = "actions"; continue
            if "risk" in l or "blocker" in l:
                cur = "risks"; continue
            if cur and ln.strip():
                sections[cur].append(ln.strip("-*• ").strip())
        return sections

def to_srt(segments: List[Dict]) -> str:
    def fmt_time(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60); ms = int((t - int(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}")
        lines.append(seg['text'])
        lines.append("")
    return "\n".join(lines)

def build_minutes_md(title: str, summary: str, items: Dict[str, List[str]]) -> str:
    md = [f"# {title}", "", "## Executive Summary", "", summary or "_(No summary)_", ""]
    if items.get("decisions"):
        md += ["## Key Decisions", ""] + [f"- {d}" for d in items["decisions"]] + [""]
    if items.get("actions"):
        md += ["## Action Items", ""] + [f"- {a}" for a in items["actions"]] + [""]
    if items.get("risks"):
        md += ["## Risks / Blockers", ""] + [f"- {r}" for r in items["risks"]] + [""]
    return "\n".join(md)

# -----------------------------
# UI
# -----------------------------
st.title("Meeting Summarizer (Free, Local Models)")
st.caption("Upload an audio/video file. We'll transcribe with faster-whisper and summarize with a free Hugging Face model.")

uploaded = st.file_uploader("Upload audio/video (wav, mp3, m4a, mp4, mkv)", type=["wav", "mp3", "m4a", "mp4", "mkv", "mov"])

col1, col2 = st.columns(2)

with col1:
    meeting_title = st.text_input("Meeting Title", value="Team Sync Meeting")
with col2:
    st.write("")
    st.write("")
    do_run = st.button("Run Summarization", type="primary", use_container_width=True)

transcript_segments = None
full_transcript = ""
summary = ""
items = {}

if do_run and uploaded:
    with st.spinner("Converting & transcribing... (this can take a while on CPU)"):
        wav_path = save_uploaded_file_to_wav(uploaded)
        segments, full_transcript, info = transcribe(wav_path, asr_model_size, device, compute_type)

    st.success(f"Transcription complete. Language: {getattr(info, 'language', 'unknown')}")

    with st.spinner("Summarizing transcript (pass 1 & 2)..."):
        summary = summarize_long_text(full_transcript, summ_model_name, max_chunk_tokens)

    with st.spinner("Extracting decisions, actions, risks..."):
        items = keyword_extract_items(full_transcript + "\\n" + summary)
        if use_optional_flan:
            ai = flan_generate_sections(full_transcript, temperature)
            # Merge unique entries
            for k in ["decisions", "actions", "risks"]:
                if k in ai:
                    merged = list(dict.fromkeys(items.get(k, []) + ai.get(k, [])))
                    items[k] = merged

    transcript_segments = segments

# -----------------------------
# Results
# -----------------------------
if transcript_segments:
    st.subheader("Executive Summary")
    st.write(summary)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Segments", len(transcript_segments))
    with c2:
        total_mins = sum(seg["end"] - seg["start"] for seg in transcript_segments) / 60.0
        st.metric("Duration (min)", f"{total_mins:.1f}")
    with c3:
        st.metric("ASR Model", asr_model_size)

    st.subheader("Structured Minutes")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Key Decisions**")
        if items.get("decisions"):
            for d in items["decisions"]:
                st.write("- " + d)
        else:
            st.write("_None detected_")
    with colB:
        st.markdown("**Action Items**")
        if items.get("actions"):
            for a in items["actions"]:
                st.write("- " + a)
        else:
            st.write("_None detected_")
    with colC:
        st.markdown("**Risks / Blockers**")
        if items.get("risks"):
            for r in items["risks"]:
                st.write("- " + r)
        else:
            st.write("_None detected_")

    st.subheader("Transcript Preview")
    with st.expander("Show transcript (first 2,000 chars)"):
        st.text(full_transcript[:2000] + ("..." if len(full_transcript) > 2000 else ""))

    # Downloads
    srt_text = to_srt(transcript_segments)
    md_text = build_minutes_md(meeting_title, summary, items)

    st.download_button("⬇ Download Transcript (SRT)", data=srt_text, file_name="transcript.srt", mime="text/plain")
    st.download_button("⬇ Download Minutes (Markdown)", data=md_text, file_name="meeting_minutes.md", mime="text/markdown")

else:
    st.info("Upload a file and click **Run Summarization** to begin.")
