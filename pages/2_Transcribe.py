"""
Transcribe Page

Upload and transcribe audio files with speaker recognition.
"""

import json
import tempfile
from pathlib import Path

import streamlit as st

from transcriber import (
    transcribe_with_speakers,
    export_transcript,
    load_profiles,
)

st.set_page_config(page_title="Transcribe", page_icon="🎤", layout="wide")

SCRIPT_DIR = Path(__file__).parent.parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"

st.title("🎤 Transcribe Audio")
st.markdown("Upload an audio file to transcribe with speaker diarization.")

st.divider()

# Settings
col1, col2 = st.columns(2)

with col1:
    model = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=2,  # default to small
        help="Larger models are more accurate but slower"
    )

with col2:
    threshold = st.slider(
        "Speaker Matching Threshold",
        0.0, 1.0, 0.7,
        help="Lower = more matches, higher = stricter matching"
    )

st.divider()

# File upload
audio_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
    help="Supported formats: MP3, WAV, M4A, OGG, FLAC, WEBM"
)

if audio_file:
    st.audio(audio_file)

    if st.button("🚀 Transcribe", type="primary", use_container_width=True):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(audio_file.name).suffix
        ) as tmp:
            tmp.write(audio_file.read())
            tmp_path = Path(tmp.name)

        try:
            # Load profiles
            profiles = load_profiles()

            # Progress indicator
            progress = st.progress(0, text="Loading model...")

            # Transcribe
            with st.spinner("Transcribing... This may take a few minutes."):
                progress.progress(20, text="Transcribing audio...")

                result = transcribe_with_speakers(
                    tmp_path,
                    profiles,
                    model_name=model,
                    threshold=threshold
                )

                progress.progress(80, text="Generating transcript...")

                # Export as JSON
                json_output = export_transcript(
                    result, Path(audio_file.name), profiles, format="json"
                )
                transcript_data = json.loads(json_output)

                # Export as markdown
                md_output = export_transcript(
                    result, Path(audio_file.name), profiles, format="markdown"
                )

                progress.progress(100, text="Done!")

            st.success("Transcription complete!")

            # Save to transcripts folder
            TRANSCRIPTS_DIR.mkdir(exist_ok=True)
            output_name = Path(audio_file.name).stem
            json_path = TRANSCRIPTS_DIR / f"{output_name}.json"
            md_path = TRANSCRIPTS_DIR / f"{output_name}.md"

            with open(json_path, "w") as f:
                f.write(json_output)
            with open(md_path, "w") as f:
                f.write(md_output)

            st.info(f"Saved to: `{json_path.name}` and `{md_path.name}`")

            st.divider()

            # Show results
            st.subheader("Transcript")

            # Display segments
            for segment in transcript_data.get("segments", []):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                speaker = segment.get("speaker_name", "Unknown")
                confidence = segment.get("confidence", 0)
                text = segment.get("text", "")

                # Format timestamp
                mins = int(start // 60)
                secs = int(start % 60)
                timestamp = f"{mins:02d}:{secs:02d}"

                # Confidence indicator
                if confidence >= 0.8:
                    conf_icon = "🟢"
                elif confidence >= 0.5:
                    conf_icon = "🟡"
                else:
                    conf_icon = "🔴"

                st.markdown(f"**[{timestamp}] {speaker}** {conf_icon}")
                st.markdown(f"> {text}")
                st.markdown("")

            st.divider()

            # Download options
            st.subheader("Download")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📄 Download Markdown",
                    md_output,
                    file_name=f"{output_name}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col2:
                st.download_button(
                    "📊 Download JSON",
                    json_output,
                    file_name=f"{output_name}.json",
                    mime="application/json",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error during transcription: {e}")
            raise e

        finally:
            tmp_path.unlink(missing_ok=True)
