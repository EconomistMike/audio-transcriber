"""
Audio Transcriber - Web GUI

A Streamlit app for managing speaker profiles and transcripts.
Run with: streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from transcriber import list_speakers, load_profiles

# Page config
st.set_page_config(
    page_title="Audio Transcriber",
    page_icon="🎙️",
    layout="wide",
)

# Paths
SCRIPT_DIR = Path(__file__).parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
AUDIO_DIR = SCRIPT_DIR / "audio"


def get_transcript_count() -> int:
    """Count JSON transcripts."""
    if not TRANSCRIPTS_DIR.exists():
        return 0
    return len(list(TRANSCRIPTS_DIR.glob("*.json")))


def get_recent_transcripts(limit: int = 5) -> list[dict]:
    """Get recent transcripts sorted by date."""
    if not TRANSCRIPTS_DIR.exists():
        return []

    transcripts = []
    for path in TRANSCRIPTS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
                transcripts.append({
                    "file": path.name,
                    "source": data.get("source", "Unknown"),
                    "date": data.get("date", "")[:10],
                    "segments": len(data.get("segments", [])),
                })
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by date descending
    transcripts.sort(key=lambda x: x["date"], reverse=True)
    return transcripts[:limit]


# Main page
st.title("🎙️ Audio Transcriber")
st.markdown("Transcribe audio with speaker diarization and automatic speaker recognition.")

st.divider()

# Stats row
col1, col2, col3 = st.columns(3)

speakers = list_speakers()
transcript_count = get_transcript_count()

with col1:
    st.metric("Enrolled Speakers", len(speakers))

with col2:
    st.metric("Transcripts", transcript_count)

with col3:
    total_samples = sum(s.get("embedding_count", 0) for s in speakers)
    st.metric("Voice Samples", total_samples)

st.divider()

# Quick actions
st.subheader("Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📁 Manage Speakers", use_container_width=True):
        st.switch_page("pages/1_Speakers.py")

with col2:
    if st.button("🎤 Transcribe Audio", use_container_width=True):
        st.switch_page("pages/2_Transcribe.py")

with col3:
    if st.button("📄 View Transcripts", use_container_width=True):
        st.switch_page("pages/3_Transcripts.py")

st.divider()

# Recent transcripts
st.subheader("Recent Transcripts")

recent = get_recent_transcripts()

if recent:
    for t in recent:
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            st.write(f"**{t['source']}**")
        with col2:
            st.write(t["date"])
        with col3:
            st.write(f"{t['segments']} segments")
else:
    st.info("No transcripts yet. Upload an audio file to get started!")

# Enrolled speakers
st.divider()
st.subheader("Enrolled Speakers")

if speakers:
    cols = st.columns(min(len(speakers), 4))
    for i, speaker in enumerate(speakers[:4]):
        with cols[i]:
            st.markdown(f"**{speaker['name']}**")
            st.caption(f"{speaker['embedding_count']} sample(s)")
else:
    st.info("No speakers enrolled. Add voice samples for automatic recognition!")
