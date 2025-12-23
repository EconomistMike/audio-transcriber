"""
Transcripts Page

View and search past transcripts.
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Transcripts", page_icon="📄", layout="wide")

SCRIPT_DIR = Path(__file__).parent.parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
AUDIO_DIR = SCRIPT_DIR / "audio"

st.title("📄 Transcripts")
st.markdown("View and search past transcripts.")

st.divider()


def load_transcripts() -> list[dict]:
    """Load all JSON transcripts."""
    if not TRANSCRIPTS_DIR.exists():
        return []

    transcripts = []
    for path in sorted(TRANSCRIPTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                data["_path"] = path
                data["_filename"] = path.stem
                transcripts.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    return transcripts


def get_unique_speakers(transcripts: list[dict]) -> list[str]:
    """Get unique speaker names across all transcripts."""
    speakers = set()
    for t in transcripts:
        for seg in t.get("segments", []):
            name = seg.get("speaker_name", "")
            if name and name != "UNKNOWN":
                speakers.add(name)
    return sorted(speakers)


# Load transcripts
transcripts = load_transcripts()

if not transcripts:
    st.info("No transcripts yet. Transcribe an audio file to get started!")
    st.stop()

# Filters
col1, col2 = st.columns(2)

with col1:
    search = st.text_input("🔍 Search", placeholder="Search transcript text...")

with col2:
    speakers = get_unique_speakers(transcripts)
    speaker_filter = st.selectbox(
        "Filter by Speaker",
        ["All Speakers"] + speakers
    )

st.divider()

# Apply filters
filtered = transcripts
if search:
    search_lower = search.lower()
    filtered = [
        t for t in filtered
        if any(search_lower in seg.get("text", "").lower() for seg in t.get("segments", []))
    ]

if speaker_filter != "All Speakers":
    filtered = [
        t for t in filtered
        if any(seg.get("speaker_name") == speaker_filter for seg in t.get("segments", []))
    ]

# Show count
st.caption(f"Showing {len(filtered)} of {len(transcripts)} transcripts")

# List transcripts
for transcript in filtered:
    source = transcript.get("source", "Unknown")
    date = transcript.get("date", "")[:10]
    duration = transcript.get("duration", 0)
    segments = transcript.get("segments", [])
    filename = transcript.get("_filename", "")

    # Duration formatting
    mins = int(duration // 60)
    secs = int(duration % 60)
    duration_str = f"{mins}:{secs:02d}"

    # Get unique speakers in this transcript
    transcript_speakers = list(set(
        seg.get("speaker_name", "Unknown")
        for seg in segments
    ))

    with st.expander(f"**{source}** - {date} ({duration_str})", expanded=False):
        # Metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", duration_str)
        with col2:
            st.metric("Segments", len(segments))
        with col3:
            st.metric("Speakers", len(transcript_speakers))

        st.markdown(f"**Speakers:** {', '.join(transcript_speakers)}")

        # Audio player (if audio file exists)
        audio_path = AUDIO_DIR / source
        if audio_path.exists():
            st.audio(str(audio_path))

        st.divider()

        # Transcript content
        for segment in segments:
            start = segment.get("start", 0)
            speaker = segment.get("speaker_name", "Unknown")
            confidence = segment.get("confidence", 0)
            text = segment.get("text", "")

            # Skip if filtering by speaker and doesn't match
            if speaker_filter != "All Speakers" and speaker != speaker_filter:
                continue

            # Skip if searching and doesn't match
            if search and search.lower() not in text.lower():
                continue

            # Format timestamp
            mins = int(start // 60)
            secs = int(start % 60)
            timestamp = f"{mins:02d}:{secs:02d}"

            # Highlight search term
            display_text = text
            if search:
                import re
                pattern = re.compile(re.escape(search), re.IGNORECASE)
                display_text = pattern.sub(f"**{search}**", text)

            # Confidence indicator
            if confidence >= 0.8:
                conf_icon = "🟢"
            elif confidence >= 0.5:
                conf_icon = "🟡"
            else:
                conf_icon = "🔴"

            st.markdown(f"**[{timestamp}] {speaker}** {conf_icon}")
            st.markdown(f"> {display_text}")

        st.divider()

        # Download options
        col1, col2, col3 = st.columns(3)

        # Read markdown version if exists
        md_path = TRANSCRIPTS_DIR / f"{filename}.md"
        json_path = transcript.get("_path")

        with col1:
            if md_path.exists():
                with open(md_path) as f:
                    md_content = f.read()
                st.download_button(
                    "📄 Markdown",
                    md_content,
                    file_name=f"{filename}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        with col2:
            if json_path:
                with open(json_path) as f:
                    json_content = f.read()
                st.download_button(
                    "📊 JSON",
                    json_content,
                    file_name=f"{filename}.json",
                    mime="application/json",
                    use_container_width=True
                )

        with col3:
            if st.button("🗑️ Delete", key=f"del_{filename}", type="secondary", use_container_width=True):
                st.session_state[f"confirm_del_{filename}"] = True

            if st.session_state.get(f"confirm_del_{filename}"):
                st.warning("Delete this transcript?")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Yes", key=f"yes_{filename}"):
                        if json_path:
                            json_path.unlink(missing_ok=True)
                        if md_path.exists():
                            md_path.unlink(missing_ok=True)
                        st.success("Deleted!")
                        st.rerun()
                with c2:
                    if st.button("No", key=f"no_{filename}"):
                        st.session_state[f"confirm_del_{filename}"] = False
                        st.rerun()
